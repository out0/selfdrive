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
from planner.physical_model import ModelCurveGenerator
import cv2
import time



class RRTNode:
    parent: 'RRTNode'
    size: float
    angle: float
    final_x: int
    final_z: int
    cost: float
    final_heading: float
    path: list[Waypoint]
    
    def __init__(self,
                 parent: 'RRTNode',
                 size: float,
                 angle: float,
                 final_x: int,
                 final_z: int,
                 cost: int,
                 final_heading: float,
                 path: list[Waypoint]
                 ) -> None:
        self.parent = parent
        self.size = size
        self.angle = angle
        self.final_x = final_x
        self.final_z = final_z
        self.cost = cost
        self.final_heading = final_heading
        self.path = path


TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_LEFT = 3
BOTTOM_RIGHT = 4

DEBUG_DUMP = True

class ComputeGraph:
    _node_list: list[RRTNode]
    _node_keys: set
    _model: ModelCurveGenerator
    _velocity_m_s: float
    
    def __init__(self, root: Waypoint, velocity_m_s: float):
        self._node_list = []
        self._node_keys = {}
        self._model = ModelCurveGenerator()
        self._velocity_m_s = velocity_m_s
        self.__add_node(
            final_x=root.x,
            final_z=root.z,
            parent_x=-1,
            parent_z=-1,
            size=0,
            angle=0,
            cost=0,
            final_heading=root.heading,
            path=None
        )
    
    def list_nodes(self) -> list[RRTNode]:
        return self._node_list
    
    def __build_key(self, x: int, z: int) -> int:
        return z * PhysicalParameters.OG_WIDTH + x
    
    def __compute_distance(self, x1: int, z1: int, x2: int, z2: int) -> float:
        dx = x2 - x1
        dz = z2 - z1
        return math.sqrt(dx ** 2 + dz ** 2)
    
    def get_node (self, final_x: int, final_z: int) -> RRTNode:
        key = self.__build_key(final_x, final_z)
        if key in self._node_keys:
            return self._node_keys[key]
        return None
    
    def get_cost(self, final_x: int, final_z: int) -> float:
        n = self.get_node(final_x, final_z)
        if n is None:
            return -1
        return n.cost
    
    def check_in_graph(self, final_x: int, final_z: int) -> bool:
        key = self.__build_key(final_x, final_z)
        return key in self._node_keys
    
    def __add_node(self, final_x: int, final_z: int, parent_x: int, parent_z: int, size: float, angle: float, cost: float, final_heading: float, path: list[Waypoint]) -> 'RRTNode':
        
        parent = None
        if parent_x >= 0 and parent_z >= 0:
            parent = self.get_node(parent_x, parent_z)
        
        node = RRTNode(
                parent,
                size,
                angle,
                final_x,
                final_z,
                cost,
                final_heading,                
                path
            )
        
        self._node_list.append(node)
        self._node_keys[self.__build_key(final_x, final_z)] = node
        return node
    

    
    def derive_node(self, og: OccupancyGrid, parent_x: int, parent_z: int, angle: float, size: float) -> RRTNode:
        path = self._model.gen_path_waypoint(
            Waypoint(parent_x, parent_z),
            self._velocity_m_s,
            angle,
            size
        )
        
        if len(path) < 2:
            return None
        
        if not og.check_all_path_feasible(path):
            return None
        
        n = path[-1]
        
        return self.__add_node(
            final_x=n.x,
            final_z=n.z,
            parent_x=parent_x,
            parent_z=parent_z,
            size=size, angle=angle,
            final_heading=n.heading,
            path=path
        )
        
    def build_link_path_between_nodes(self, og: OccupancyGrid, start: RRTNode, end: RRTNode) -> list[Waypoint]:
        path = self._model.connect_nodes_with_path(
            Waypoint(start.final_x, start.final_z, start.final_heading),
            Waypoint(end.final_x, end.final_z, end.final_heading),
            self._velocity_m_s
        )
        
        if len(path) < 2:
            return None
        
        if not og.check_all_path_feasible(path):
            return None
        
        return path
        
    def find_nearest_neighbor(self, x: int, z: int):
        best_dist = 99999999
        best_node = None
        for p in self._node_list:
            dist = self.__compute_distance(p.final_x, p.final_z, x, z)
            if dist < best_dist:
                best_dist = dist
                best_node = p
        return best_node
            
    def optimize_graph(self, og: OccupancyGrid, node: RRTNode, search_radius: float):        
        for target in self._node_list:
            self.__optimize_node(og, node, target, search_radius)

   
    def __optimize_node(self, og: OccupancyGrid, node: RRTNode, target: RRTNode, search_radius: float) -> None:
        dx = abs(target.x - node.x)
        dz = abs(target.z - node.z)
        
        if dz == 0:   # pure left or right neighbors are not desired
            return
        
        if node.final_z >= target.final_z:  #lets not optimize by pointing backwards
            return
        
        dist_to_target = math.sqrt(dx * dx + dz * dz)
        if (dist_to_target > search_radius):
            return
        
        link_path = self.build_link_path_between_nodes(og, target, node)
        
        path_size = len(link_path) # rough approximation
        

        if target.parent is None:
            target_parent_x = -1
            target_parent_z = -1
        else:
            target_parent_x = target.parent.final_x
            target_parent_z = target.parent.final_z

        # I'm searching my parent, which is forbidden because it is a cyclic ref.
        if target_parent_x == node.x and\
            target_parent_z == node.z:
                return
            
        target_cost = target.cost
        my_cost = node.cost

        if (my_cost <= target_cost + path_size):
            return
        
        # perform the optimization change
        node.parent = target
        node.cost = target_cost + path_size
    
    def get_parent(self, x: int, z: int) -> RRTNode:
        n = self.get_node(x, z)
        if n is None:
            return None
        return n.parent
       
       
    def find_best_neighbor(self, goal: Waypoint, reach_dist: float) -> RRTNode:
        best_neighbor = None
        best_heading_err = 99999999
        
        for node in self._node_list:
            dist = self.__compute_distance(goal.x, goal.z, node.x, node.z) 
            heading_err = abs(goal.heading - node.heading)
            if dist > reach_dist:
                continue
            
            if best_heading_err > heading_err:
                best_heading_err = heading_err
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

    
    def __check_link(self, parent_x: int, parent_z: int, new_node: Waypoint) -> bool:       
        path = self._build_path(parent_x, parent_z, new_node)
        if path is None:
            return
        return self._og.check_all_path_feasible(path, False)
 
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
       
    def __generate_informed_random_node(self) -> RRTNode:
        
        
        
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
            
        new_node = self.__generate_node_towards_point(p_x, p_z, x_rand, z_rand)
        if new_node is None:
            return None

        if self._og.check_waypoint_class_is_obstacle(new_node.x, new_node.z):
            return None
        
        if not self.__check_link(p_x, p_z, new_node):
            return None
        
        cost = self._compute_graph.get_cost(p_x, p_z)
        if cost < 0:
            return None
        
        return self._compute_graph.add_point(new_node.x, new_node.z, new_node.heading, p_x, p_z, cost + self.__compute_distance(new_node.x, new_node.z, p_x, p_z))

        
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
            self._goal_result.start.heading,
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
                
                path = self._build_path(p_x, p_z, Waypoint(x, z, p.heading))
                if path is None:
                    return
                for p in path:
                    frame[p.z, p.x, :] = [255, 255, 255]
                    
        if planned_path is not None:
            path = self.__interpolate_sparse_path(planned_path)
            for p in path:
                    frame[p.z, p.x, :] = [0, 0, 255]
                    
        cv2.imwrite("debug_rrt.png", frame)
        
    def _build_path(self, parent_x: int, parent_z: int, new_node: Waypoint) -> list[Waypoint]:
        x = new_node.x
        z = new_node.z
        heading = new_node.heading
        
        dx = abs(x - parent_x)
        dz = abs(z - parent_z)
        num_steps = max(dx , dz)
        
        
        #heading = Waypoint.compute_heading(Waypoint(parent_x, parent_z), Waypoint(x, z))
        
        
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

        first = self._compute_graph.find_best_neighbor(self._goal_result.goal,  RRTPlanner.REACH_DIST)
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
            path = self._build_path(parent.x, parent.z, sparse_path[i])
            res.append(Waypoint(parent.x, parent.z))
            res.extend(path)
            res.append(sparse_path[i])
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