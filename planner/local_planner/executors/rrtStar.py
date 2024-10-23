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
from planner.goal_point_discover import GoalPointDiscoverResult
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
import cv2

class RRTNode:
    parent: 'RRTNode'
    x: int
    z: int
    cost: float
    
    def __init__(self, x: int, z: int, parent: 'RRTNode') -> None:
        self.x = x
        self.z = z
        self.parent = parent
        self.cost = 0
        
    def compute_node_dist(n1: 'RRTNode', x: int, z: int) -> float:
        dx = x - n1.x
        dz = z - n1.z
        return math.sqrt(dx ** 2 + dz ** 2)


class RRTPlanner (LocalPathPlannerExecutor):
    _search: bool
    _plan_task: Thread
    _result: PlanningResult
    _og: OccupancyGrid
    _planning_data: PlanningData
    _post_plan_smooth: bool
    _width: int
    _height: int
    _max_step: float
    _node_list: list[RRTNode]
    
    NAME = "RRT*"
    REACH_DISTANCE = 15
    DEBUG = False
    
    def __init__(self,  
                 max_exec_time_ms: int, 
                 max_steps: float
                 ) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._goal_result = None
        self._max_step = max_steps
        
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        
        self._planning_data = planner_data
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self._node_list = []
        self._search = True
        self._goal_result = goal_result
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
        
    def __generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
    
    def __find_nearest_node(self, x: int, z: int) -> RRTNode:
        best_dist = sys.maxsize
        nearest = None
        for n in self._node_list:
            dist = RRTNode.compute_node_dist(n, x, z)
            if dist < best_dist:
                best_dist = dist
                nearest = n
        return nearest
    
    
    def __generate_node_towards_point(self, base_point: RRTNode, x_rand: int, z_rand: int) -> RRTNode:
        dist = RRTNode.compute_node_dist(base_point, x_rand, z_rand)
        
        if dist < self._max_step:
            return RRTNode(x_rand, z_rand, None)
        
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point.z, x_rand - base_point.x)
        x = base_point.x + math.floor(self._max_step * math.cos(slope))
        z = base_point.z + math.floor(self._max_step * math.sin(slope))
        return RRTNode(x, z, None)
    
    def __check_path_feasible(self, p1: RRTNode, p2: RRTNode) -> bool:
        dx = abs(p2.x - p1.x)
        dz = abs(p2.z - p1.z)
        steps = max(dz, dx)
        
        if steps == 0:
            return False
        
        x_inc = dx / steps
        z_inc = dz / steps
        
        x = p1.x
        z = p1.z

        path = []
        
        for _ in range(steps):
            x += x_inc
            z += z_inc
            path.append(Waypoint(x,z))

        return self._og.check_all_path_feasible(path, True)
    
    def __perform_planning(self) -> None:
        #self._result.planner_name = RRTPlanner.NAME
        self.set_exec_started()
        self._node_list.append(RRTNode(self._goal_result.start.x, self._goal_result.start.z, None))
        
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            if self._check_timeout() or not self._search:
                loop_search = False
                continue
            
            x_rand, z_rand = self.__generate_random_point()
            
            nearest = self.__find_nearest_node(x_rand, z_rand)
            
            new_node: RRTNode = self.__generate_node_towards_point(nearest, x_rand, z_rand)
            
            if not self.__check_path_feasible(nearest, new_node):
                continue
            
            near_nodes: list[RRTNode] = []
            for n in self._node_list:
                if RRTNode.compute_node_dist(n, new_node.x, new_node.z) < self._max_step:
                    near_nodes.append(n)
            
            best_cost_parent = nearest
            best_cost = RRTNode.compute_node_dist(nearest, new_node.x, new_node.z) + best_cost_parent.cost
            
            for n in near_nodes:
                cost = RRTNode.compute_node_dist(new_node, n.x, n.z)
                if cost < best_cost and self.__check_path_feasible(n, new_node):
                    best_cost = cost
                    best_cost_parent = n
            
            new_node.parent = best_cost_parent
            new_node.cost = best_cost
            
            if RRTNode.compute_node_dist(new_node,
                                        self._goal_result.goal.x,
                                        self._goal_result.goal.z
            ) <= RRTPlanner.REACH_DISTANCE:
                loop_search = False
            
            self._node_list.append(new_node)
            
        if RRTPlanner.DEBUG:
            self.__debug_step(self._node_list)
        
        # finding the path
        
        path: list[Waypoint] = []
        current = self.__find_nearest_node(self._goal_result.goal.x, self._goal_result.goal.z)
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        result_type = PlannerResultType.INVALID_PATH
        
            
        if len(path) > 1:
            path = WaypointInterpolator.path_smooth_rebuild(path)
            if self._og.check_all_path_feasible(path):
                result_type = PlannerResultType.VALID
               
        self._result = PlanningResult(
            planner_name = RRTPlanner.NAME,
            ego_location = self._planning_data.ego_location,
            goal = self._planning_data.goal,
            next_goal = self._planning_data.next_goal,
            local_start = self._goal_result.start,
            local_goal = self._goal_result.goal,
            direction = self._goal_result.direction,
            timeout = False,
            path = path,
            result_type = result_type,
            total_exec_time_ms = self.get_execution_time()
        )
        
        self._search = False


    def __debug_step(self, node_list: list[RRTNode]):
        
        frame = self._og.get_color_frame()
        start = self._goal_result.start
        
        
        for n in node_list:
            
            if n.parent is None:
                x, z = start.x, start.z
            else:
                x, z = n.parent.x, n.parent.z
                
            path = WaypointInterpolator.interpolate_straight_line_path2(
                Waypoint(x, z), 
                Waypoint(n.x, n.z),
                PhysicalParameters.OG_WIDTH,
                PhysicalParameters.OG_HEIGHT,
                30)
            
            for p in path:
                frame[p.z, p.x, :] = [255, 255, 255]
            
        cv2.imwrite("debug_rrt.png", frame)