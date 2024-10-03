from model.waypoint import Waypoint
from model.map_pose import MapPose
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.physical_model import ModelCurveGenerator
from queue import PriorityQueue
from .debug_dump import dump_result
from planner.goal_point_discover import GoalPointDiscoverResult

MAX_FLOAT = 100000000

DIR_TOP = 0
DIR_TOP_LEFT = 1
DIR_TOP_RIGHT = 2
DIR_LEFT = 3
DIR_RIGHT = 4
DIR_BOTTOM_LEFT = 5
DIR_BOTTOM = 6
DIR_BOTTOM_RIGHT = 7

TURNING_COST = 5


DEBUG_DUMP = False

# SAFE_W = 10
# SAFE_H = 20

# def draw_safe_square(frame: np.ndarray, pos: tuple[int, int], angle: float):
#     r = angle
#     c = math.cos(r)
#     s = math.sin(r)
#     for z in range(-SAFE_H, SAFE_H + 1):
#         for x in range(-SAFE_W, SAFE_W + 1):
#             (xo, zo) = pos
#             xl = round(x * c - z * s) + xo
#             zl = round(x * s + z * c) + zo
#             frame[zl, xl, :] = [255, 255, 0]
#     frame[pos[1], pos[0], :] = [0, 0, 255]


class Node:
    cost: float
    dist: float
    #absolute_dir: GridDirection
    relative_dir: int
    local_pose: Waypoint
    global_pose: MapPose
    parent: 'Node'
    
    def __init__(self,
                 parent: 'Node',
                 local_pose: Waypoint,
                 global_pose: MapPose,
                 cost: float, 
                 dist: float,
                 #abs_dir: GridDirection, 
                 rel_dir: int):
        
        self.parent = parent
        self.local_pose = local_pose
        self.cost = cost
        self.dist = dist
        #self.absolute_dir = abs_dir
        self.relative_dir = rel_dir
        self.global_pose = global_pose

    def __gt__(self, other):
        return self.cost > other.dist

    def __lt__(self, other):
        return self.cost < other.dist

    def __eq__(self, other):
        return self.cost == other.dist

    
F_CURRENT_BEST_GUESS = 0
G_CHEAPEST_COST_TO_PATH = 1
H_DISTANCE_TO_GOAL = 2


class HybridAStarPlanner (LocalPathPlannerExecutor):
    NAME = "Hybrid A*"
    _search: bool
    _plan_task: Thread
    _og: OccupancyGrid
    _result: PlanningResult
    _post_plan_smooth: bool
    _minimal_width: int
    _minimal_height: int
    _ego_lower_bound: Waypoint
    _ego_upper_bound: Waypoint
    _map_converter: CoordinateConverter
    _planner_data: PlanningData
    _model_curve_gen: ModelCurveGenerator
    _expected_velocity_meters_s: float
    
    def __init__(self, 
                 max_exec_time_ms: int,
                 map_converter: CoordinateConverter,
                 expected_velocity_meters_s: float) -> None:
        
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._map_converter = map_converter
        self._planner_data = None
        self._model_curve_gen = ModelCurveGenerator()
        self._expected_velocity_meters_s = expected_velocity_meters_s
    
    
    def __create_node(self, parent: Node, point: Waypoint, pose: MapPose, rel_dir: int, cost: float) -> Node:
        
        return Node(
            local_pose=point,
            global_pose=pose,
            rel_dir=rel_dir,
            cost=cost,
            dist = self._og.get_frame()[point.z, point.x, 1],
            parent=parent
        )
    
    def __find_first_unfeasible(self, path : list[Waypoint]) -> int:
        chk = self._og.check_path_feasible(path)
        for i in range(len(path)):
            if (path[i].x < 0 or path[i].x >= self._og.width()):
                continue
            
            if (path[i].z < 0 or path[i].z >= self._og.height()):
                continue
            
            if not chk[i]:
                return i
        return len(path)
            
   
    def __build_child_nodes(self, parent: Node, path: list[MapPose], rel_dir: int) -> list[Node]:
        path_w = self._map_converter.convert_map_path_to_waypoint(self._planner_data.ego_location, path)
        
        # if DEBUG_OUTP:
        #     self.show_path(path_w, color=[0, 0, 255])
        
        first_unfeasible = self.__find_first_unfeasible(path_w)
       
        if first_unfeasible == 0:
            return []

        # if DEBUG_OUTP:
        #     self.show_path(path_w[:first_unfeasible], color=[255, 0, 0])
        
        res = []

        cost = parent.cost
        if rel_dir != 0:
            cost += TURNING_COST

        local_cost = cost
        f = self._og.get_frame()
        point_parent = parent
        for i in range(first_unfeasible):
            point = path_w[i]
            
            if point.x < 0 or point.x >= self._og.width():
                continue
            
            if point.z < 0 or point.z >= self._og.height():
                continue
            
            local_cost += f[point.z, point.x, 1]
            n = self.__create_node(
                parent=point_parent,
                point=point,
                pose= path[i],
                rel_dir=-1, 
                cost=cost)

            res.append(n)
            point_parent = n
            

        return res
        


    def _compute_free_surroundings(self,
                                   node: Node) -> list[Node]:
        res = []

        tl, t, tr = self._model_curve_gen.gen_possible_top_paths(node.global_pose, self._expected_velocity_meters_s)       
        res.extend(self.__build_child_nodes(node, t, 0))        
        res.extend(self.__build_child_nodes(node, tl, -1))
        res.extend(self.__build_child_nodes(node, tr, 1))       
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
        #self._og = None
        

    
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult):
        self._search = True
        self._planner_data = planner_data
        self._goal_result = goal_result
        self._result = None
        self._og = planner_data.og
        
        if goal_result.goal is None:
            self._result = PlanningResult.build_basic_response_data(
                HybridAStarPlanner.NAME,
                PlannerResultType.INVALID_GOAL,
                planner_data,
                goal_result
            )
            self._search = False
            return
        
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()
    

    def get_result(self) -> PlanningResult:
        return self._result
    
    def _add_points(self, p1: Waypoint, p2: Waypoint) -> Waypoint:
        return Waypoint(p1.x + p2.x, p1.z + p2.z)

    def __perform_planning(self) -> None:
        self.set_exec_started()
        self._search = True
        
        root = self.__create_node(
            point=self._goal_result.start,
            parent=None, 
            pose=self._planner_data.ego_location, 
            rel_dir=0, 
            cost=0)
        
        open_list = PriorityQueue()
        open_list.put((root.dist, root))

        best_candidate: Node = None
        best_distance_to_goal: float = MAX_FLOAT
        closed: dict[str, Node] = {}

        perform_search = self._search
        while self._search and perform_search and not open_list.empty():
            _, curr_point = open_list.get(block=False)

            # if DEBUG_DUMP:
            #     result = PlanningResult(
            #         planner_name = HybridAStarPlanner.NAME,
            #         ego_location = self._planner_data.ego_location,
            #         goal = self._planner_data.goal,
            #         next_goal = self._planner_data.next_goal,
            #         local_start = self._goal_result.start,
            #         local_goal = self._goal_result.goal,
            #         direction = self._goal_result.direction,
            #         timeout = False,
            #         path = path,
            #         result_type = PlannerResultType.VALID,
            #         total_exec_time_ms = self.get_execution_time()
            #     )
            #     dump_result(self._og, result)

            if self._check_timeout():
                perform_search = False
                continue
            
            if best_distance_to_goal > curr_point.dist:
                best_distance_to_goal = curr_point.dist
                best_candidate = curr_point
            
            if curr_point.dist <= 4:
                best_candidate = curr_point
                best_distance_to_goal = 0
                perform_search = False
                continue
            
            if curr_point.local_pose.x == self._goal_result.goal.x and curr_point.local_pose.z == self._goal_result.goal.z:
                best_candidate = curr_point
                best_distance_to_goal = 0
                perform_search = False
                continue
            
            key = str(curr_point.local_pose)
            
            if key in closed:
                continue

            closed[key] = curr_point

            free_surroundings = self._compute_free_surroundings(
                curr_point)
       
            for node in free_surroundings:
                k = str(node.local_pose)

                if k in closed:
                    if node.cost < closed[k].cost:
                        closed[k] = node
                    
                else:
                    open_list.put((node.dist, node))
                    
            
        if best_candidate is None:
            self._result = PlanningResult.build_basic_response_data(
                HybridAStarPlanner.NAME,
                PlannerResultType.INVALID_PATH,
                self._planner_data,
                self._goal_result
            )            
            self._search = False
            return

        path: list[Waypoint] = []
        p = best_candidate
        
        while p is not None and p is not root:
            path.append(p.local_pose)
            p = p.parent
    
        #path.append(self._result.local_start)
        path.reverse()

        self._result = PlanningResult(
            planner_name = HybridAStarPlanner.NAME,
            ego_location = self._planner_data.ego_location,
            goal = self._planner_data.goal,
            next_goal = self._planner_data.next_goal,
            local_start = self._goal_result.start,
            local_goal = self._goal_result.goal,
            direction = self._goal_result.direction,
            timeout = False,
            path = path,
            result_type = PlannerResultType.VALID,
            total_exec_time_ms = self.get_execution_time()
        )
        self._search = False
        
        if DEBUG_DUMP:
           dump_result(self._og, self._result)

