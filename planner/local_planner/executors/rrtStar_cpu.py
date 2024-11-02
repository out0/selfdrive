from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
import math, numpy as np
from planner.local_planner.executors.dubins_curves import Dubins
import random, sys
from vision.occupancy_grid_cuda import OccupancyGrid
from .waypoint_interpolator import WaypointInterpolator
from model.physical_parameters import PhysicalParameters
from planner.goal_point_discover import GoalPointDiscoverResult, GridDirection
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
import cv2
import time


class RRTNode:
    parent: 'RRTNode'
    x: int
    z: int
    cost: float
    
    def __init__(self, x: int, z: int, parent: 'RRTNode', cost = 0) -> None:
        self.x = x
        self.z = z
        self.parent = parent
        self.cost = cost
        
    def compute_node_dist(n1: 'RRTNode', x: int, z: int) -> float:
        dx = x - n1.x
        dz = z - n1.z
        return math.sqrt(dx ** 2 + dz ** 2)

TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_LEFT = 3
BOTTOM_RIGHT = 4

DEBUG_DUMP = False

class ComputeGraph:
    _node_list: list[RRTNode]
    _node_keys: set
    
    def __init__(self):
        self._node_list = []
        self._node_keys = {}
    
    def list_nodes(self) -> list[RRTNode]:
        return self._node_list
    
    def __build_key(self, x: int, z: int) -> int:
        return z * PhysicalParameters.OG_WIDTH + x
    
    def __compute_distance(self, x1: int, z1: int, x2: int, z2: int) -> float:
        dx = x2 - x1
        dz = z2 - z1
        return math.sqrt(dx ** 2 + dz ** 2)
    
    def get_node (self, x: int, z: int) -> RRTNode:
        key = self.__build_key(x, z)
        if key in self._node_keys:
            return self._node_keys[key]
        return None
    
    def get_cost(self, x: int, z: int) -> float:
        n = self.get_node(x, z)
        if n is None:
            return -1
        return n.cost
    
    def check_in_graph(self, x: int, z: int) -> bool:
        key = self.__build_key(x, z)
        return key in self._node_keys
    
    def add_point(self, x: int, z: int, parent_x: int, parent_z: int, cost: float) -> None:
        
        parent = None
        if parent_x >= 0 and parent_z >= 0:
            parent = self.get_node(parent_x, parent_z)
        
        node = RRTNode(
                x,
                z,
                parent,
                cost
            )
        
        self._node_list.append(node)
        self._node_keys[self.__build_key(x, z)] = node
        return node
        
    def find_nearest_neighbor(self, x: int, z: int):
        best_dist = 99999999
        best_node = None
        for p in self._node_list:
            dist = self.__compute_distance(p.x, p.z, x, z)
            if dist < best_dist:
                best_dist = dist
                best_node = p
        return best_node
            
    def optimize_graph(self, og: OccupancyGrid, node: RRTNode, search_radius: float):
        # PAREI
        
        for target in self._node_list:
            self.__optimize_node(og, node, target, search_radius)

    def __compute_heading(p1_x: int, p1_z: int, p2_x: int, p2_z: int, width: int, height: int) -> tuple[float, bool]:
        
        if (p1_x == p2_x and p1_z == p2_z):
            return 0.0, False

        if (p1_x < 0 or p1_z < 0 or p2_x < 0 or p2_z < 0):
            return 0.0, False

        if (p1_x >= width or p1_z >= height or p2_x >= width or p2_z >= height):
            return 0.0, False

        dx = p2_x - p1_x
        dz = p2_z - p1_z
        heading = 3.141592654 / 2 - math.atan2(-dz, dx)
        if (heading > 3.141592654): # greater than 180 deg
            heading = heading - 2 * 3.141592654

        return heading, True

    def __check_link_between_p1_p2(self, og: OccupancyGrid, p1_x: int, p1_z: int, p2_x: int, p2_z: int) -> bool:
        dx = abs(p2_x - p1_x)
        dz = abs(p2_z - p1_z)
        
        num_steps = math.floor(max(dx, dz))
        if (num_steps == 0):
            return False

        dxs = dx / num_steps
        dzs = dz / num_steps

        last_x = p2_x
        last_z = p2_z
        
        heading, valid = ComputeGraph.__compute_heading(
            p1_x, 
            p1_z, 
            p2_x, 
            p2_z, 
            PhysicalParameters.OG_WIDTH, 
            PhysicalParameters.OG_HEIGHT)
        
        if not valid:
            return False

        if (p2_x < p1_x):
            dxs = -dxs
            
        if (p2_z < p1_z):        
            dzs = -dzs
        
        path = []
        last_x = p1_x
        last_z = p1_z
        for i in range(1, num_steps + 1):
            px = p1_x + i * dxs
            pz = p2_x + i * dzs
            if px == last_x and pz == last_z:
                continue
            path.append(Waypoint(px, pz, heading))
            last_x = px
            last_z = pz
        
        return og.check_all_path_feasible(path)
        
        

    def __optimize_node(self, og: OccupancyGrid, node: RRTNode, target: RRTNode, search_radius: float) -> None:
        dx = abs(target.x - node.x)
        dz = abs(target.z - node.z)
        
        if dz == 0:   # pure left or right neighbors are not desired
            return
        
        dist_to_target = math.sqrt(dx * dx + dz * dz);
        if (dist_to_target > search_radius):
            return

        if target.parent is None:
            target_parent_x = -1
            target_parent_z = -1
        else:
            target_parent_x = target.parent.x
            target_parent_z = target.parent.z

        # I'm searching my parent, which is forbidden because it is a cyclic ref.
        if target_parent_x == node.x and\
            target_parent_z == node.z:
                return
            
        target_cost = target.cost
        my_cost = node.cost

        if (my_cost <= target_cost + dist_to_target):
            return

        # lets check if I can connect to target
        if not self.__check_link_between_p1_p2(og, node.x, node.z, target.x, target.z):
            return
        
        # perform the optimization change
        node.parent = target
        node.cost = target_cost + dist_to_target
    
    def get_parent(self, x: int, z: int) -> RRTNode:
        n = self.get_node(x, z)
        if n is None:
            return None
        return n.parent
       
    def find_best_neighbor(self, goal_x: int, goal_z: int,  reach_dist: float) -> RRTNode:
        best_neighbor = None
        best_cost = 99999999
        
        for node in self._node_list:
            dist = self.__compute_distance(goal_x, goal_z, node.x, node.z)
            if dist > reach_dist:
                continue
            
            cost_if_node_is_selected = dist + node.cost
            if best_cost > cost_if_node_is_selected:
                best_cost = cost_if_node_is_selected
                best_neighbor = node
        
        return best_neighbor
    


class RRTPlanner (LocalPathPlannerExecutor):
    _search: bool
    _plan_task: Thread
    _result: PlanningResult
    _og: OccupancyGrid
    __found_goal: bool
    _planning_data: PlanningData
    _post_plan_smooth: bool
    _width: int
    _height: int
    _compute_graph: ComputeGraph
    __rough_direction_lim1: Waypoint
    __rough_direction_lim2: Waypoint
    __rough_direction_lim1: Waypoint
    __rough_direction_lim2: Waypoint

    
    NAME = "RRT*"
    RRT_STEP = 40
    SEARCH_RADIUS = 30
    REACH_DIST = 10
    
    def __init__(self,  
                  max_exec_time_ms: int, reasonable_exec_time_ms: int = 50
                 ) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._goal_result = None
        self.__found_goal = False
        self._reasonable_exec_time_ms = reasonable_exec_time_ms
        self._compute_graph = ComputeGraph()
        
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:        
        self.__planner_data = planner_data
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self._search = True
        self._goal_result = goal_result
        self.__rough_direction_lim1, self.__rough_direction_lim2, self.__direction_lim1, self.__direction_lim2 = self.__compute_direction_limits(goal_result)
        self._og.set_goal_vectorized(goal_result.goal)
        self._plan_task = Thread(target=self._perform_planning)
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

    def __generate_node_towards_point(self, base_point_x: int, base_point_z: int, x_rand: int, z_rand: int) -> tuple[int, int]:
        dist = self.__compute_distance(base_point_x, base_point_z, x_rand, z_rand)
        
        if dist < RRTPlanner.RRT_STEP:
            return x_rand, z_rand
               
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point_z, x_rand - base_point_x)
        x = base_point_x + math.floor(RRTPlanner.RRT_STEP * math.cos(slope))
        z = base_point_z + math.floor(RRTPlanner.RRT_STEP * math.sin(slope))
        return x, z

    def __check_link(self, parent_x: int, parent_z: int, x: int, z: int) -> bool:       
        path = self._build_path(parent_x, parent_z, x, z)
        if path is None:
            return
        return self._og.check_all_path_feasible(path, False)
 
    # def __compute_direction_limits(self, res: GoalPointDiscoverResult) -> tuple[Waypoint, Waypoint, Waypoint, Waypoint]:
    #     if res.start.x > res.goal.x:
    #         if res.start.z > res.goal.z:
    #             #TOP_LEFT
    #             return  Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
    #                     Waypoint(0, 0), Waypoint(res.start.x, res.start.z)            
    #         else:
    #             #BOTTOM_RIGHT
    #             return  Waypoint(0, res.start.z + 1), Waypoint(PhysicalParameters.OG_WIDTH - 1, PhysicalParameters.OG_HEIGHT - 1),\
    #                     Waypoint(res.start.x, res.start.z), Waypoint(PhysicalParameters.OG_WIDTH -1 , PhysicalParameters.OG_HEIGHT - 1)
    #     else:
    #         if res.start.z > res.goal.z:
    #             #TOP_RIGHT
    #             return  Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
    #                     Waypoint(res.start.x, 0), Waypoint(PhysicalParameters.OG_WIDTH -1 , res.start.z)
    #         else:
    #             #BOTTOM_LEFT
    #             return  Waypoint(0, res.start.z + 1), Waypoint(PhysicalParameters.OG_WIDTH - 1, PhysicalParameters.OG_HEIGHT - 1),\
    #                     Waypoint(0, res.start.z + 1), Waypoint(res.start.x, PhysicalParameters.OG_HEIGHT - 1)   

    def __compute_direction_limits(self, res: GoalPointDiscoverResult) -> tuple[Waypoint, Waypoint, Waypoint, Waypoint]:
        if res.start.x > res.goal.x:
            #TOP_LEFT
            return Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
                Waypoint(0, 0), Waypoint(res.start.x, res.start.z)            
        else:
            #TOP_RIGHT
            return Waypoint(0, 0), Waypoint(PhysicalParameters.OG_WIDTH - 1, res.start.z - 1),\
                Waypoint(res.start.x, 0), Waypoint(PhysicalParameters.OG_WIDTH -1 , res.start.z)

    
    def __generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
       
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
    
    def __add_new_node_to_graph(self) -> RRTNode:
        x_rand, z_rand = self.__generate_informed_random_point()
            
        if self._compute_graph.check_in_graph(x_rand, z_rand):
            return None
            
        nearest = self._compute_graph.find_nearest_neighbor(x_rand, z_rand)
            
        if nearest is None:
            return None
            
        p_x, p_z = nearest.x, nearest.z
            
        (new_x, new_z) = self.__generate_node_towards_point(p_x, p_z, x_rand, z_rand)

        if self._og.check_waypoint_class_is_obstacle(new_x, new_z):
            return None
        
        if not self.__check_link(p_x, p_z, new_x, new_z):
            return None
        
        cost = self._compute_graph.get_cost(p_x, p_z)
        if cost < 0:
            return None
        
        return self._compute_graph.add_point(new_x, new_z, p_x, p_z, cost + self.__compute_distance(new_x, new_z, p_x, p_z))

        
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
        
            self._compute_graph.optimize_graph(self._og, node, RRTPlanner.SEARCH_RADIUS)
                                             
            if not self.__found_goal:
                dist_to_goal = self._og.get_cost(node.x, node.z)
                
                if dist_to_goal <= RRTPlanner.REACH_DIST:
                    self.__found_goal = True
                    return False
            
            else:
                if 1000*(time.time() - start) >= min_exec_time_ms:
                   return False
               
    def _perform_planning(self) -> None:
        
        self.set_exec_started()
        
        self._compute_graph.add_point(
            self._goal_result.start.x, 
            self._goal_result.start.z,
            -1,
            -1,
            0
        )
       
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            timeout = self.__search(self._reasonable_exec_time_ms)

            if timeout and not self.__found_goal:
                self._result = self.__build_timeout_no_path_response()
                self._search = False
                return
            
            first_path = self._build_sparse_path_from_process_result()
            if first_path is None:
                continue
            
            path = self.__post_process_smooth(first_path)
            
            if DEBUG_DUMP:
                self.dump_result(first_path)

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

              
    def dump_result(self, planned_path: list[Waypoint] = None):
        frame = self._og.get_color_frame()
        
        for p in self._compute_graph.list_nodes():
            x = math.floor(p.x)
            z = math.floor(p.z)
            if p.parent is None:
                p_x = -1
                p_z = -1
            else:
                p_x = math.floor(p.parent.x)
                p_z = math.floor(p.parent.z)
        
            if p_x < 0 or p_z < 0:
                frame[z, x, :] = [255, 255, 255]
            else:
                path = self._build_path(p_x, p_z, x, z)
                if path is None:
                    return
                for p in path:
                    frame[p.z, p.x, :] = [255, 255, 255]
                    
        if planned_path is not None:
            path = self.__interpolate_sparse_path(planned_path)
            for p in path:
                    frame[p.z, p.x, :] = [0, 0, 255]
                    
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
        
    def _build_sparse_path_from_process_result(self) -> list[Waypoint]:
        
        # radius = RRTPlanner.REACH_DIST
        # first = None
        # while first is None and radius <= PhysicalParameters.OG_HEIGHT:
        #     first = self._compute_frame.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z, radius)
        #     radius = 2 * radius

        first = self._compute_graph.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z,  RRTPlanner.REACH_DIST)
        if first is None:
            return None

        #first = self._compute_frame.find_nearest_neighbor(self._goal_result.goal.x, self._goal_result.goal.z)
        
        sparse_path: list[Waypoint] = []
        current = first

        while current is not None:
            x, z = current.x, current.z
            if x < 0 or z < 0:
                break
            
            sparse_path.append(Waypoint(x, z))
            
            current = self._compute_graph.get_parent(x, z)
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

    def __post_process_smooth(self, sparse_path: list[Waypoint]):
        
        path = self.__interpolate_sparse_path(sparse_path)
        
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

        path.reverse()
        return path
            
        # if self._og.check_all_path_feasible(path):
        #      return path

        # return None       