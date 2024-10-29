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
        
    NAME = "RRT"
    
    RRT_STEP = 30
    SEARCH_RADIUS = 30
    REACH_DIST = 10
    
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._compute_frame = CudaGraph(PhysicalParameters.OG_WIDTH, PhysicalParameters.OG_HEIGHT)
        self.__planner_data = None
        self._node_list = []
        self._num_nodes_in_list = 0
        self._start = Waypoint(128, PhysicalParameters.EGO_UPPER_BOUND.z - 1)
        self._og = None
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self.__planner_data = planner_data
        self._search = True
        self._num_nodes_in_list = 0
        self._goal_result = goal_result
        self.__rough_direction_lim1, self.__rough_direction_lim2, self.__direction_lim1, self.__direction_lim2 = self.__compute_direction_limits(goal_result)
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
                path = self.__build_path(p_x, p_z, x, z)
                for p in path:
                    frame[p.z, p.x, :] = [255, 255, 255]
            
            
            
            
        cv2.imwrite("debug_rrt.png", frame)
   
    def __build_path(self, parent_x: int, parent_z: int, x: int, z: int) -> list[Waypoint]:
        dx = abs(x - parent_x)
        dz = abs(z - parent_z)
        num_steps = max(dx , dz)
        
        if num_steps == 0:
            return None
        
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
                Waypoint(x, z)
            )
        return path

    def __check_link(self, parent_x: int, parent_z: int, x: int, z: int) -> bool:       
        path = self.__build_path(parent_x, parent_z, x, z)
        return self._og.check_all_path_feasible(path)

    

        
    
    def __link(self, parent_x: int, parent_z: int, new_x: int, new_z: int) -> bool:
       
        base_cost = self._compute_frame.get_cost(parent_x, parent_z)
            
        if not self.__check_link(parent_x, parent_z, new_x, new_z):
            return False

        self._compute_frame.add_point(new_x, new_z, parent_x, parent_z, base_cost + self.__compute_distance(new_x, new_z, parent_x, parent_z))
                
        return True
    
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
        self._compute_frame.add_point(new_x, new_z, p_x, p_z, cost)
        return RRTNode(new_x, new_z, p_x, p_z, cost)
    
    #def __optimize_graph_with_node(self, node: RRTNode) -> None:
    
    
    def __perform_planning(self) -> None:
        #self._result.planner_name = RRTPlanner.NAME
        self.set_exec_started()
        
        root_cost = self._og.get_cost(self._start.x, self._start.z)        
        self._compute_frame.add_point(self._start.x, self._start.z, -1, -1, root_cost)
        
        loop_search = self._search
        self._rst_timeout()
        
        found_goal = False
        
        while loop_search:
            if self._check_timeout():
                loop_search = False
                continue
            
            node = self.__add_new_node_to_graph()
            
            if node is None :
                continue
            
            if DEBUG_DUMP:
                self.dump_result()
            
            self._compute_frame.optimize_graph(node.x, node.z, node.parent_x, node.parent_z, node.cost, RRTPlanner.SEARCH_RADIUS)
            
            # self.__optimize_graph_with_node(node)
            
            # best_parent = self._compute_frame.find_best_neighbor(new_x, new_z, RRTPlanner.SEARCH_RADIUS)
            
            # if best_parent is None:
            #     continue
            
            # parent_x, parent_z = best_parent
            
            # if not self.__link(parent_x, parent_z, new_x, new_z):
            #     continue
            
            if DEBUG_DUMP:
                self.dump_result()

            if not found_goal:
                dist_to_goal = self._og.get_cost(node.x, node.z)
                
                if dist_to_goal <= RRTPlanner.REACH_DIST:
                    found_goal = True
            
            else:
                if self._check_half_timeout() and found_goal:
                    loop_search = False
            
        
        # finding the path
        
        radius = RRTPlanner.SEARCH_RADIUS
        current = None
        while current is None and radius <= PhysicalParameters.OG_HEIGHT:
            current = self._compute_frame.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z, radius)
            radius = 2 * radius
        
#        current = self._compute_frame.find_nearest_neighbor(self._goal_result.goal.x, self._goal_result.goal.z)
        
        if current is None:
            return PlanningResult(
                planner_name = RRTPlanner.NAME,
                ego_location = self.__planner_data.ego_location,
                goal = self.__planner_data.goal,
                next_goal = self.__planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = None,
                result_type = PlannerResultType.INVALID_PATH,
                total_exec_time_ms = self.get_execution_time()
        )
        
        sparse_path: list[Waypoint] = []
        
        
        while current is not None:
            x, z = current
            if x < 0 or z < 0:
                break
            
            sparse_path.append(Waypoint(x, z))
            
            current = self._compute_frame.get_parent(x, z)
            if current is None:
                break
        
        full_path: list[Waypoint] = []
        
        parent = sparse_path[0]
        
        for i in range(1, len(sparse_path)):
            x, z = sparse_path[i].x, sparse_path[i].z
            path = self.__build_path(parent.x, parent.z, x, z)
            full_path.append(Waypoint(parent.x, parent.z))
            full_path.extend(path)
            full_path.append(Waypoint(x, z))

        
        path.reverse()
        result_type = PlannerResultType.INVALID_PATH
        result_type = PlannerResultType.VALID

        if len(path) > 1:
            s_try = [30, 20, 10, 1]
            
            for s in s_try:
                smooth_path = WaypointInterpolator.path_smooth_rebuild(path, s)
                if len(smooth_path) == 0:
                    continue
                if self._og.check_all_path_feasible(smooth_path):
                    result_type = PlannerResultType.VALID
                    path = smooth_path
                    break
            
            if result_type == PlannerResultType.INVALID_PATH:
                if (len(path) >= 10):
                    if self._og.check_all_path_feasible(path):
                        result_type = PlannerResultType.VALID

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
            result_type = result_type,
            total_exec_time_ms = self.get_execution_time()
        )
        
        self._search = False