from model.waypoint import Waypoint
from model.map_pose import MapPose
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
import sys, math, numpy as np
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.physical_model import ModelCurveGenerator
import cv2
from queue import PriorityQueue

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


DEBUG_OUTP = False

SAFE_W = 10
SAFE_H = 20

def draw_safe_square(frame: np.ndarray, pos: tuple[int, int], angle: float):
    r = angle
    c = math.cos(r)
    s = math.sin(r)
    for z in range(-SAFE_H, SAFE_H + 1):
        for x in range(-SAFE_W, SAFE_W + 1):
            (xo, zo) = pos
            xl = round(x * c - z * s) + xo
            zl = round(x * s + z * c) + zo
            frame[zl, xl, :] = [255, 255, 0]
    frame[pos[1], pos[0], :] = [0, 0, 255]


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
        
        if DEBUG_OUTP:
            self.show_path(path_w, color=[0, 0, 255])
        
        first_unfeasible = self.__find_first_unfeasible(path_w)
       
        if first_unfeasible == 0:
            return []

        if DEBUG_OUTP:
            self.show_path(path_w[:first_unfeasible], color=[255, 0, 0])
        
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
        
    def show_path (self, path: list[Waypoint],  color = [255, 255, 255]):
        if (len(path) == 0): return
        frame = np.array(cv2.imread("plan_debug_outp.png"))

        for p in path:
            # angle = math.atan2(p.z - before.z, p.x - before.x)+  math.radians(-90)
            # draw_safe_square(frame, (p.x, p.z), angle) 
            if p.x < 0 or p.x >= frame.shape[1]:
                continue
            if p.z < 0 or p.z >= frame.shape[0]:
                continue
            
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
        self._result = partial_result.clone()
        self._og = planner_data.og
        
        #self._og.set_goal_vectorized(planner_data.local_goal)
        
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
        
        root = self.__create_node(
            point=self._result.local_start,
            parent=None, 
            pose=self._result.ego_location, 
            rel_dir=0, 
            cost=0)
        
        open_list = PriorityQueue()
        open_list.put((root.dist, root))

        best_candidate: Node = None
        best_distance_to_goal: float = MAX_FLOAT
        closed: dict[str, Node] = {}

        if DEBUG_OUTP:
            frame = self._og.get_color_frame()
            cv2.imwrite("plan_debug_outp.png", frame)
            self.show_point(self._result.local_goal, color=[255, 0, 0])


        perform_search = self._search
        while perform_search and not open_list.empty():
            _, curr_point = open_list.get(block=False)

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
            
            if curr_point.local_pose.x == self._result.local_goal.x and curr_point.local_pose.z == self._result.local_goal.z:
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
            self._result.result_type = PlannerResultType.INVALID_GOAL
            self._search = False
            return

        path: list[Waypoint] = []
        p = best_candidate
        
        while p is not None and p is not root:
            path.append(p.local_pose)
            p = p.parent
    
        #path.append(self._result.local_start)
        path.reverse()

        self._result.path = path
        self._result.result_type = PlannerResultType.VALID
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False

