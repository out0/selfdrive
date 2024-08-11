from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
from vision.occupancy_grid_cuda import GridDirection
import numpy as np, sys, cv2
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

MOVING_COST = [
    1,  # up
    1,  # diag up
    1,  # diag up
    50,  # side
    50,  # side
    500,  # down!
    500,  # down!
    500  # down!
]
# MOVING_COST = [
#     1,  # up
#     5,  # diag up
#     5,  # diag up
#     20,  # side
#     20,  # side
#     500,  # down!
#     500,  # down!
#     10000  # down!
# ]



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


class VectorialAStarPlanner (LocalPathPlannerExecutor):
    NAME = "A*"
    _search: bool
    _plan_task: Thread
    _planner_data: PlanningData
    _result: PlanningResult
    _post_plan_smooth: bool
    _minimal_width: int
    _minimal_height: int
    _ego_lower_bound: Waypoint
    _ego_upper_bound: Waypoint
    
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None


    def __checkTraversable(self, current: Waypoint, next: Waypoint, direction: int) -> bool:
        if current.z < 0 or current.z >= self._og.height():
            return False
        if current.x < 0 or current.x >= self._og.width():
            return False
        if next.z < 0 or next.z >= self._og.height():
                return False
        if next.x < 0 or next.x >= self._og.width():
            return False
        
        return self._og.check_direction_allowed(current.x, current.z, direction)

    def _compute_free_surroundings(self, point: Waypoint) -> list[bool]:
        res: list[bool] = [False, False, False, False, False, False, False, False]
        
        res[DIR_TOP] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP]), GridDirection.HEADING_0.value)
        res[DIR_TOP_LEFT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_LEFT]), GridDirection.HEADING_0.value | GridDirection.HEADING_45.value)
        res[DIR_TOP_RIGHT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_RIGHT]), GridDirection.HEADING_0.value | GridDirection.HEADING_MINUS_45.value)
        res[DIR_LEFT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_LEFT]), GridDirection.HEADING_90.value)
        res[DIR_RIGHT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_RIGHT]), GridDirection.HEADING_90.value)
        res[DIR_BOTTOM_LEFT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_LEFT]), GridDirection.HEADING_0.value | GridDirection.HEADING_MINUS_45.value)
        res[DIR_BOTTOM] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM]), GridDirection.HEADING_0.value)
        res[DIR_BOTTOM_RIGHT] = self.__checkTraversable(point, self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_RIGHT]), GridDirection.HEADING_0.value | GridDirection.HEADING_45.value)
        return res

    def set_bounds(self, lower_bound: Waypoint, upper_bound: Waypoint):
        self._ego_lower_bound = lower_bound
        self._ego_upper_bound = upper_bound

    def is_planning(self) -> bool:
        return self._search
       

    def cancel(self) -> None:
        self._search = False

        if self._plan_task is not None and self._plan_task.is_alive:
            self._plan_task.join()

        self._plan_task = None
        self._og = None
        
    def show_point (self, p: Waypoint,  color = [255, 255, 255]):
        frame = cv2.imread("plan_debug_outp.png")
    
        if p.x < 0:
            p.x = 0
        if p.z < 0:
            p.z = 0
        
        if p.x > 0:
            frame[p.z, p.x - 1, :] = color
        if p.x < frame.shape[1] - 1:
            frame[p.z, p.x + 1, :] = color
        if p.z > 0:
            frame[p.z - 1, p.x, :] = color
        if p.z < frame.shape[0] - 1:
            frame[p.z + 1, p.x, :] = color
    
    
        frame[p.z, p.x, :] = color
        cv2.imwrite("plan_debug_outp.png", frame)

    def plan(self, planner_data: PlanningData, partial_result: PlanningResult):
        
        self._search = True
        self._planner_data = planner_data
        self._result = partial_result        
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()
    

    def get_result(self) -> PlanningResult:
        return self._result
    
    def _add_points(self, p1: Waypoint, p2: Waypoint) -> Waypoint:
        return Waypoint(p1.x + p2.x, p1.z + p2.z)

    def __perform_planning(self) -> None:
        self.set_exec_started()
        self._search = True
        
        if self._result.local_goal is None:
            self._result.total_exec_time_ms = self.get_execution_time()
            self._result.result_type = PlannerResultType.INVALID_GOAL
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
        # cv2.imwrite("plan_debug_outp.png", frame)
        # self.show_point(self._result.local_goal, color=[255, 0, 0])

        perform_search = self._search
        while perform_search and not open_list.empty():
            curr_point = open_list.get(block=False)

            if self._check_timeout():
                perform_search = False
                continue

            plan_grid.set_closed(curr_point)

            free_surroundings = self._compute_free_surroundings(
                curr_point)

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

                # self.show_point(next_point)
                # time.sleep(0.1)
                
                distance_to_goal = frame[next_point.z, next_point.x, 1]

                if distance_to_goal < best_distance_to_goal:
                    best_distance_to_goal = distance_to_goal
                    best_possible = next_point
                
                if next_point.x == self._result.local_goal.x and next_point.z == self._result.local_goal.z:
                    best_possible = next_point
                    best_distance_to_goal = 0
                    perform_search = False

                g = curr_costs[G_CHEAPEST_COST_TO_PATH] + MOVING_COST[dir]
                h = distance_to_goal
                f = g + h

                next_costs = plan_grid.get_costs(next_point)

                if next_costs[F_CURRENT_BEST_GUESS] > f:
                    plan_grid.set_costs(next_point, [f, g, h])
                    plan_grid.set_parent(next_point, curr_point)
                    open_list.put(QueuedPoint(next_point, f))

        if best_possible is None:
            self._result.result_type = PlannerResultType.INVALID_GOAL
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

