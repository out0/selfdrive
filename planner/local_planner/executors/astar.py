from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
import math, numpy as np, sys
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
from queue import PriorityQueue


MAX_FLOAT = sys.float_info.max - 10

DIR_TOP = 0
DIR_TOP_LEFT = 1
DIR_TOP_RIGHT = 2
DIR_LEFT = 3
DIR_RIGHT = 4
DIR_BOTTOM_LEFT = 5
DIR_BOTTOM = 6
DIR_BOTTOM_RIGHT = 7


COMPUTE_DIRECTION_POS = [
    Waypoint(0, -1),
    Waypoint(-1, -1),
    Waypoint(1, -1),
    Waypoint(-1, 0),
    Waypoint(1, 0),
    Waypoint(0, 1),
    Waypoint(-1, 1),
    Waypoint(1, 1),
]

# MOVING_COST = [
#     1,  # up
#     1,  # diag up
#     1,  # diag up
#     1,  # side
#     1,  # side
#     1,  # down!
#     1,  # down!
#     1  # down!
# ]
MOVING_COST = [
    1,  # up
    5,  # diag up
    5,  # diag up
    20,  # side
    20,  # side
    500,  # down!
    500,  # down!
    10000  # down!
]



class QueuedPoint(Waypoint):
    cost: float

    def __init__(self, p: Waypoint, cost: float):
        super().__init__(p.x, p.z)
        self.cost = cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost


class NpPlanGrid:
    grid: np.array
    costs: np.array

    def __init__(self, width: int, height: int):
        self.grid = np.full((height, width, 3), -1)
        self.costs = np.full((height, width, 3), MAX_FLOAT)

    def set_closed(self, point: Waypoint) -> None:
        self.grid[point.z, point.x, 0] = 1

    def is_closed(self, point: Waypoint) -> bool:
        return self.grid[point.z, point.x, 0] == 1

    def set_parent(self, point: Waypoint, parent: Waypoint) -> None:
        self.grid[point.z, point.x, 1] = parent.x
        self.grid[point.z, point.x, 2] = parent.z

    def set_parent_by_coord(self, point: Waypoint, coord: list[int]) -> None:
        self.grid[point.z, point.x, 1] = coord[0]
        self.grid[point.z, point.x, 2] = coord[1]


    def get_costs(self, point: Waypoint) -> np.array:
        return self.costs[point.z, point.x]
    
    def set_costs(self, point: Waypoint, lst: list) -> None:
        self.costs[point.z, point.x] = lst

    def get_parent(self, point: Waypoint) -> Waypoint:
        return Waypoint(self.grid[point.z, point.x][1], self.grid[point.z, point.x][2])
    
F_CURRENT_BEST_GUESS = 0
G_CHEAPEST_COST_TO_PATH = 1
H_DISTANCE_TO_GOAL = 2


class AStarPlanner (LocalPathPlannerExecutor):
    NAME = "A*"
    _search: bool
    _plan_task: Thread
    _planner_data: PlanningData
    _result: PlanningResult
    _post_plan_smooth: bool
    _og: OccupancyGrid
    
    def __init__(self, 
                 max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None


    def _checkObstacle(self, frame: np.array, point: Waypoint) -> bool:
        if point.z < 0 or point.z >= frame.shape[0]:
            return True
        if point.x < 0 or point.x >= frame.shape[1]:
            return True
        return frame[point.z][point.x][2] == 0
    
    def _add_points(self, p1: Waypoint, p2: Waypoint) -> Waypoint:
        return Waypoint(p1.x + p2.x, p1.z + p2.z)

    def _compute_free_surroundings(self,
                                   frame: np.array,
                                   point: Waypoint) -> list[bool]:
        res: list[bool] = []

        for _ in range(0, 8):
            res.append(False)

        res[DIR_TOP] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP])
        )
        res[DIR_TOP_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_LEFT])
        )
        res[DIR_TOP_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_RIGHT])
        )
        res[DIR_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_LEFT])
        )
        res[DIR_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_RIGHT])
        )                
        res[DIR_BOTTOM_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_LEFT])
        )
        res[DIR_BOTTOM] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM]))
        res[DIR_BOTTOM_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_RIGHT])
        )

        return res

    def _compute_euclidian_distance(self, p1: Waypoint, p2: Waypoint) -> float:
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.sqrt(dz * dz + dx * dx)

    def is_planning(self) -> bool:
        return self._search
       

    def cancel(self) -> None:
        self._search = False

        if self._plan_task is not None and self._plan_task.is_alive:
            self._plan_task.join()

        self._plan_task = None
        self._og = None

    def plan(self, planner_data: PlanningData, partial_result: PlanningResult):
        
        self._search = True
        self._planner_data = planner_data
        self._result = partial_result
        
        self._og = self._planner_data.og.clone()
        
        self._og.set_goal(partial_result.local_goal)
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()
    

    def get_result(self) -> PlanningResult:
        return self._result

    def __perform_planning(self) -> None:
        self.set_exec_started()
        self._search = True
        
        if self._result.local_goal is None:
            self._result.result_type = PlannerResultType.INVALID_GOAL
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return

        plan_grid = NpPlanGrid(self._og.width(), self._og.height())
        
        plan_grid.set_costs(self._result.local_start, [0, 0, 0])
        #plan_grid.set_parent_by_coord(start, [-1, -1])

        open_list = PriorityQueue()
        open_list.put(QueuedPoint(self._result.local_start, 0))

        best_possible = None
        best_distance_to_goal: float = MAX_FLOAT

        frame = self._og.get_frame()

        perform_search = self._search
        while perform_search and not open_list.empty():
            curr_point = open_list.get(block=False)

            if self._check_timeout():
                perform_search = False
                continue

            plan_grid.set_closed(curr_point)

            free_surroundings = self._compute_free_surroundings(
                frame, curr_point)

            f: float = MAX_FLOAT
            g: float = MAX_FLOAT
            h: float = MAX_FLOAT

            curr_costs = plan_grid.get_costs(curr_point)

            for dir in range(0, 6):
                if not free_surroundings[dir]:
                    continue

                next_point = self._add_points(
                    curr_point, COMPUTE_DIRECTION_POS[dir])
                
                if plan_grid.is_closed(next_point):
                    continue

                distance_to_goal = frame[next_point.z, next_point.x, 1]

                if distance_to_goal < best_distance_to_goal:
                    best_distance_to_goal = distance_to_goal
                    best_possible = next_point
                
                if next_point.x == self._result.local_goal.x and next_point.z == self._result.local_goal.z:
                    best_possible = next_point
                    best_distance_to_goal = 0
                    perform_search = False

                g = curr_costs[G_CHEAPEST_COST_TO_PATH] + MOVING_COST[dir] - frame[next_point.z, next_point.x, 2]
                h = distance_to_goal
                f = g + h

                next_costs = plan_grid.get_costs(next_point)

                if next_costs[F_CURRENT_BEST_GUESS] > f:
                    plan_grid.set_costs(next_point, [f, g, h])
                    plan_grid.set_parent(next_point, curr_point)
                    open_list.put(QueuedPoint(next_point, f))

        if best_possible is None:
            self._result.result_type = PlannerResultType.INVALID_GOAL
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return

        path: list[Waypoint] = []

        path.append(best_possible)
        
        p = best_possible        
        parent = plan_grid.get_parent(p)
        while parent.x >= 0:
            path.append(parent)
            p = parent
            parent = plan_grid.get_parent(p)

        path.append(self._result.local_start)
        path.reverse()

        self._result.path = path
        self._result.result_type = PlannerResultType.VALID
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False

