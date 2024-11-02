from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
import math, numpy as np
from planner.local_planner.executors.dubins_curves import Dubins
import random, sys
from model.physical_parameters import PhysicalParameters
from utils.cuda_rrt_accel.cuda_rrt_accel import CudaGraph
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.goal_point_discover import GoalPointDiscoverResult
import cv2
import time
#from .debug_dump import dump_result

DEBUG_DUMP = False


class RRTNode:
    x: int
    z: int
    parent_x: int
    parent_z: int
    cost: float
    
    def __init__(self,
                 x: int,
                 z: int,
                 parent_x: int,
                 parent_z: int,
                 cost: float):
        self.x = x
        self.z = z
        self.parent_x = parent_x
        self.parent_z = parent_z
        self.cost = cost


class RRTPlanner (LocalPathPlannerExecutor):
    _search: bool
    _plan_task: Thread
    _compute_frame: CudaGraph
    _og: OccupancyGrid
    _result: PlanningResult
    _start: Waypoint
    _plnning_data: PlanningData
    _post_plan_smooth: bool
    _width: int
    _height: int
    _max_step: float
    _goal_result: GoalPointDiscoverResult
    _node_list: list
    __direction_lim1: Waypoint
    __direction_lim2: Waypoint
    __rough_direction_lim1: Waypoint
    __rough_direction_lim2: Waypoint
    __planner_data: PlanningData
    __found_goal: bool
        
    NAME = "RRT"
    
    RRT_STEP = 30
    SEARCH_RADIUS = 20
    REACH_DIST = 10
    
    
    
    def __init__(self, max_exec_time_ms: int, reasonable_exec_time_ms: int = 50) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._compute_frame = CudaGraph(
            PhysicalParameters.OG_WIDTH, 
            PhysicalParameters.OG_HEIGHT,
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND)
        
        self.__planner_data = None
        self._node_list = []
        self._num_nodes_in_list = 0
        self._start = Waypoint(128, PhysicalParameters.EGO_UPPER_BOUND.z - 1)
        self._og = None
        self._reasonable_exec_time_ms = reasonable_exec_time_ms
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self.__planner_data = planner_data
        self._search = True
        self._num_nodes_in_list = 0
        self._goal_result = goal_result
        self.__rough_direction_lim1, self.__rough_direction_lim2, self.__direction_lim1, self.__direction_lim2 = self.__compute_direction_limits(goal_result)
        self.__found_goal = False
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()

    def cancel(self) -> None:
        self._search = False

        if self._plan_task is not None and self._plan_task.is_alive:
            self._plan_task.join()

        self._plan_task = None
        self._og = None

    def is_planning(self) -> bool:
        return self._search
       

    def get_result(self) -> PlanningResult:
        return self._result


    def __compute_distance(self, x1: int, z1: int, x2: int, z2: int) -> float:
        dx = x2 - x1
        dz = z2 - z1
        return math.sqrt(dx ** 2 + dz ** 2)
        
    def __generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
    
    def __compute_direction_limits(self, res: GoalPointDiscoverResult) -> tuple[Waypoint, Waypoint, Waypoint, Waypoint]:
        if res.start.x > res.goal.x:
            if res.start.z > res.goal.z:
                #TOP_LEFT
                return  Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
                        Waypoint(0, 0), Waypoint(res.start.x, res.start.z)            
            else:
                #BOTTOM_RIGHT
                return  Waypoint(0, res.start.z + 1), Waypoint(PhysicalParameters.OG_WIDTH - 1, PhysicalParameters.OG_HEIGHT - 1),\
                        Waypoint(res.start.x, res.start.z), Waypoint(PhysicalParameters.OG_WIDTH -1 , PhysicalParameters.OG_HEIGHT - 1)
        else:
            if res.start.z > res.goal.z:
                #TOP_RIGHT
                return  Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
                        Waypoint(res.start.x, 0), Waypoint(PhysicalParameters.OG_WIDTH -1 , res.start.z)
            else:
                #BOTTOM_LEFT
                return  Waypoint(0, res.start.z + 1), Waypoint(PhysicalParameters.OG_WIDTH - 1, PhysicalParameters.OG_HEIGHT - 1),\
                        Waypoint(0, res.start.z + 1), Waypoint(res.start.x, PhysicalParameters.OG_HEIGHT - 1)
       

    def __generate_informed_random_point(self) -> tuple[int, int]:
        
        p = random.randint(0, 9)
        
        if p == 0: # 10% chance of going total random
            return self.__generate_random_point()
        
        if p >= 1 and p <= 3: # 25% chance of going rough direction
            w = random.randint(self.__rough_direction_lim1.x, self.__rough_direction_lim2.x)
            h = random.randint(self.__rough_direction_lim1.z, self.__rough_direction_lim2.z)
        else:
            w = random.randint(self.__direction_lim1.x, self.__direction_lim2.x)
            h = random.randint(self.__direction_lim1.z, self.__direction_lim2.z)

        return (w, h)    
    
    
    def __generate_node_towards_point(self, base_point_x: int, base_point_z: int, x_rand: int, z_rand: int) -> tuple[int, int]:
        dist = self.__compute_distance(base_point_x, base_point_z, x_rand, z_rand)
        
        if dist < RRTPlanner.RRT_STEP:
            return x_rand, z_rand
               
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point_z, x_rand - base_point_x)
        x = base_point_x + math.floor(RRTPlanner.RRT_STEP * math.cos(slope))
        z = base_point_z + math.floor(RRTPlanner.RRT_STEP * math.sin(slope))
        return x, z


    def dump_result(self):
        frame = self._og.get_color_frame()
        node_list = self._compute_frame.list_nodes()
        
        for i in range(0, node_list.shape[0]):
            x = math.floor(node_list[i, 0])
            z = math.floor(node_list[i, 1])
            p_x = math.floor(node_list[i, 2])
            p_z = math.floor(node_list[i, 3])
            if p_x < 0 or p_z < 0:
                frame[z, x, :] = [255, 255, 255]
            else:
                path = self._build_path(p_x, p_z, x, z)
                if path is None:
                    return
                for p in path:
                    frame[p.z, p.x, :] = [255, 255, 255]
        cv2.imwrite("debug_rrt.png", frame)
   
    def _build_path(self, parent_x: int, parent_z: int, x: int, z: int) -> list[Waypoint]:
        dx = abs(x - parent_x)
        dz = abs(z - parent_z)
        num_steps = max(dx , dz)
        
        
        heading = Waypoint.compute_heading(Waypoint(parent_x, parent_z), Waypoint(x, z))
        
        # if num_steps < 5:
        #     return None
        
        dxs = dx / num_steps
        dzs = dz / num_steps
        
        last_x = parent_x
        last_z = parent_z
        
        path: list[Waypoint] = []
        
        if x < parent_x:
            dxs = -dxs
        if z < parent_z:
            dzs = -dzs

        for i in range(1, num_steps):            
            delta_x = dxs * i
            delta_z = dzs * i
            
            x = math.floor(parent_x + delta_x)
            z = math.floor(parent_z + delta_z)
            
            if x == last_x and z == last_z:
                continue
            
            path.append(
                Waypoint(x, z, heading)
            )
        return path

    def __check_link(self, parent_x: int, parent_z: int, x: int, z: int) -> bool:       
        path = self._build_path(parent_x, parent_z, x, z)
        if path is None:
            return
        return self._og.check_all_path_feasible(path, False)

    
    
    def __add_new_node_to_graph(self) -> RRTNode:
        x_rand, z_rand = self.__generate_informed_random_point()
            
        if self._compute_frame.check_in_graph(x_rand, z_rand):
            return None
            
        nearest = self._compute_frame.find_nearest_neighbor(x_rand, z_rand)
            
        if nearest is None:
            return None
            
        p_x, p_z = nearest
            
        (new_x, new_z) = self.__generate_node_towards_point(p_x, p_z, x_rand, z_rand)

        if self._og.check_waypoint_class_is_obstacle(new_x, new_z):
            return None
        
        if not self.__check_link(p_x, p_z, new_x, new_z):
            return None
        
        cost = self._compute_frame.get_cost(p_x, p_z)            
        self._compute_frame.add_point(new_x, new_z, p_x, p_z, cost + self.__compute_distance(new_x, new_z, p_x, p_z))
        return RRTNode(new_x, new_z, p_x, p_z, cost)
    

    def __search(self, min_exec_time_ms: int) -> bool:
        start = time.time()
        while True:
            if self._check_timeout():
                return True
            
            node = self.__add_new_node_to_graph()
            
            if node is None :
                continue
        
            if DEBUG_DUMP:
                self.dump_result()
        
            self._compute_frame.optimize_graph(self._og.get_cuda_frame(), node.x, node.z, node.parent_x, node.parent_z, node.cost, RRTPlanner.SEARCH_RADIUS)
            
            if DEBUG_DUMP:
                self.dump_result()
                                  
            if not self.__found_goal:
                dist_to_goal = self._og.get_cost(node.x, node.z)
                
                if dist_to_goal <= RRTPlanner.REACH_DIST:
                    self.__found_goal = True
                    return False
            
            else:
                if 1000*(time.time() - start) >= min_exec_time_ms:
                   return False
    
    def __build_timeout_no_path_response(self) -> PlanningResult:
        return PlanningResult(
                planner_name = RRTPlanner.NAME,
                ego_location = self.__planner_data.ego_location,
                goal = self.__planner_data.goal,
                next_goal = self.__planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = True,
                path = None,
                result_type = PlannerResultType.INVALID_PATH,
                total_exec_time_ms = self.get_execution_time())
        

        
    def __perform_planning(self) -> None:
        #self._result.planner_name = RRTPlanner.NAME
        self.set_exec_started()
        
        #root_cost = self._og.get_cost(self._start.x, self._start.z)
        root_cost = 0
        self._compute_frame.add_point(self._start.x, self._start.z, -1, -1, root_cost)
        
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            timeout = self.__search(self._reasonable_exec_time_ms)
            
            if DEBUG_DUMP:
                self.dump_result()
            
            
            if timeout and not self.__found_goal:
                self._result = self.__build_timeout_no_path_response()
                self._search = False
                return
            
            first_path = self._build_sparse_path_from_process_result()
            if first_path is None:
                continue
            
            path = self._post_process_smooth(first_path)
            

            if path is not None:
                self._result = PlanningResult(
                    planner_name = RRTPlanner.NAME,
                    ego_location = self.__planner_data.ego_location,
                    goal = self.__planner_data.goal,
                    next_goal = self.__planner_data.next_goal,
                    local_start = self._goal_result.start,
                    local_goal = self._goal_result.goal,
                    direction = self._goal_result.direction,
                    timeout = False,
                    path = path,
                    result_type = PlannerResultType.VALID,
                    total_exec_time_ms = self.get_execution_time()
                )
                self._search = False
                return
                
            
            if timeout:
                self._result = self.__build_timeout_no_path_response()
                self._search = False
                
       
    
    def _build_sparse_path_from_process_result(self) -> list[Waypoint]:
        
        # radius = RRTPlanner.REACH_DIST
        # first = None
        # while first is None and radius <= PhysicalParameters.OG_HEIGHT:
        #     first = self._compute_frame.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z, radius)
        #     radius = 2 * radius

        first = self._compute_frame.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z,  RRTPlanner.REACH_DIST)
        if first is None:
            return None

        #first = self._compute_frame.find_nearest_neighbor(self._goal_result.goal.x, self._goal_result.goal.z)
        
        sparse_path: list[Waypoint] = []
        current = first

        while current is not None:
            x, z = current
            if x < 0 or z < 0:
                break
            
            sparse_path.append(Waypoint(x, z))
            
            current = self._compute_frame.get_parent(x, z)
            if current is None:
                break
        return sparse_path
    
    def __interpolate_path(self, sparse_path: list[Waypoint]) -> list[Waypoint]:
        parent = sparse_path[0]
        res = []
        
        for i in range(1, len(sparse_path)):
            x, z = sparse_path[i].x, sparse_path[i].z
            path = self._build_path(parent.x, parent.z, x, z)
            res.append(Waypoint(parent.x, parent.z))
            res.extend(path)
            res.append(Waypoint(x, z))
            parent = sparse_path[i]
        #res.reverse()
        return res
    
    def __interpolate_sparse_path(self, sparse_path: list[Waypoint]) -> list[Waypoint]:
        path = None
        if (len(sparse_path) >= 5):
            try:
                sparse_path.reverse()
                smooth_path = WaypointInterpolator.path_smooth_rebuild(sparse_path, 1)
                if self._og.check_all_path_feasible(smooth_path):
                    return smooth_path
                else:
                    sparse_path.reverse()
            except:
                path = None
                sparse_path.reverse()
        
        path = self.__interpolate_path(sparse_path)
        if len(path) < 5:
            return None
        return path
    
    def _post_process_smooth(self, sparse_path: list[Waypoint]):
        
        path = self.__interpolate_sparse_path(sparse_path)
        return path
        
        s_try = [30, 20, 10, 1]
            
        for s in s_try:
            try:
                smooth_path = WaypointInterpolator.path_smooth_rebuild(path, s)
                if len(smooth_path) == 0:
                    continue

                if self._og.check_all_path_feasible(smooth_path):
                    return smooth_path
            except:
                continue
            
        if self._og.check_all_path_feasible(path):
             return path

        return None                   