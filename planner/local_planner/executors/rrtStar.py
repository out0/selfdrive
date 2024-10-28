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
    
    __direction_lim1: Waypoint
    __direction_lim2: Waypoint

    __rough_direction_lim1: Waypoint
    __rough_direction_lim2: Waypoint

    
    NAME = "RRT*"
    REACH_DISTANCE = 15
    DEBUG = False
    
    def __init__(self,  
                 max_exec_time_ms: int, 
                 max_steps: float
                 ) -> None:
        super().__init__(-1)
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
        self.__rough_direction_lim1, self.__rough_direction_lim2, self.__direction_lim1, self.__direction_lim2 = self._compute_direction_limits(goal_result)
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
        
    def _generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
    
    def _compute_direction_limits(self, res: GoalPointDiscoverResult) -> tuple[Waypoint, Waypoint, Waypoint, Waypoint]:
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
       

    def _generate_informed_random_point(self) -> tuple[int, int]:
        
        p = random.randint(0, 9)
        
        if p == 0: # 10% chance of going total random
            return self._generate_random_point()
        
        if p >= 1 and p <= 3: # 25% chance of going rough direction
            w = random.randint(self.__rough_direction_lim1.x, self.__rough_direction_lim2.x)
            h = random.randint(self.__rough_direction_lim1.z, self.__rough_direction_lim2.z)
        else:
            w = random.randint(self.__direction_lim1.x, self.__direction_lim2.x)
            h = random.randint(self.__direction_lim1.z, self.__direction_lim2.z)

        return (w, h)
    
    def _find_nearest_node(self, x: int, z: int) -> RRTNode:
        best_dist = sys.maxsize
        nearest = None
        for n in self._node_list:
            dist = RRTNode.compute_node_dist(n, x, z)
            if dist < best_dist:
                best_dist = dist
                nearest = n
        return nearest
    
    
    def _generate_node_towards_point(self, base_point: RRTNode, x_rand: int, z_rand: int) -> RRTNode:
        dist = RRTNode.compute_node_dist(base_point, x_rand, z_rand)
        
        if dist < self._max_step:
            return RRTNode(x_rand, z_rand, None)
        
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point.z, x_rand - base_point.x)
        x = base_point.x + math.floor(self._max_step * math.cos(slope))
        z = base_point.z + math.floor(self._max_step * math.sin(slope))
        return RRTNode(x, z, None)
    
    
    def _build_link_if_feasible(self, p1:RRTNode, p2:RRTNode, path_size: float) -> list[RRTNode]:
        
        steps = min(math.floor(path_size), 30)
        if steps <= 0:
            return None
        
        path: list[Waypoint] = WaypointInterpolator.interpolate_straight_line_path2(
            Waypoint(p1.x, p1.z),
            Waypoint(p2.x, p2.z),
            PhysicalParameters.OG_WIDTH,
            PhysicalParameters.OG_HEIGHT,
            steps
        )
        
        if not self._og.check_all_path_feasible(path):
            return None
        
        link = []
        parent = p1
        for p in path:
            cost = self._og.get_cost(p.x, p.z)
            n = RRTNode(
                p.x,
                p.z,
                parent,
                cost
            )
            link.append(n)
            parent = n
        
        return link

    def _new_random_node(self) -> RRTNode:
        x_rand, z_rand = self._generate_informed_random_point()
        #nearest = self.__find_nearest_node(x_rand, z_rand)
        return RRTNode(x_rand, z_rand, None)
          
    
    def _build_best_link_possible(self, node: RRTNode) -> list[RRTNode]:
        best_parent = None
        best_cost = 9999999999
        link = None
        
        for p in self._node_list:
            path_size = RRTNode.compute_node_dist(node, p.x, p.z)
            if path_size + p.cost < best_cost:
                link = self._build_link_if_feasible(p, node, path_size)
                if link is None:
                    continue
                best_cost = path_size + p.cost
                best_parent = p
        
        if best_parent == None:
            return None
        
        return link
        
        
                
    
    def _perform_planning(self) -> None:
        
        self.set_exec_started()
        
        root = RRTNode(
            self._goal_result.start.x, 
            self._goal_result.start.z, 
            None,
            cost=self._og.get_cost(
                self._goal_result.start.x, 
                self._goal_result.start.z, 
            ))
        self._node_list.append(root)
        
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            if self._check_timeout() or not self._search:
                loop_search = False
                continue
            
            random_node = self._new_random_node()
            
            if (self._og.check_waypoint_class_is_obstacle(random_node.x, random_node.z)):
                continue
            
            # if self.__check_link_feasible(random_node):
            #     continue
            
            link = self._build_best_link_possible(random_node)
            
            if link is None:
                continue
            
            self._node_list.extend(link)

            if RRTNode.compute_node_dist(link[-1],
                                        self._goal_result.goal.x,
                                        self._goal_result.goal.z
            ) <= RRTPlanner.REACH_DISTANCE:
                loop_search = False
           
            if RRTPlanner.DEBUG:
                self._debug_step(self._node_list)

        
        # finding the path
        
        path: list[Waypoint] = []
        current = self._find_nearest_node(self._goal_result.goal.x, self._goal_result.goal.z)
        
        if current is None:
            return PlanningResult(
                planner_name = RRTPlanner.NAME,
                ego_location = self._planning_data.ego_location,
                goal = self._planning_data.goal,
                next_goal = self._planning_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = None,
                result_type = PlannerResultType.INVALID_PATH,
                total_exec_time_ms = self.get_execution_time()
        )
        
        while current is not None:
            path.append(Waypoint(current.x, current.z))
            current = current.parent
        
        path.reverse()
        result_type = PlannerResultType.INVALID_PATH
        
            
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
        
        if (len(self._result.path) == 0):
            z = 1
        
        self._search = False


    def _debug_step(self, node_list: list[RRTNode]):
        
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
        
    def _dump_path(self, path: list[Waypoint]):
        
        frame = self._og.get_color_frame()

        for p in path:
            frame[p.z, p.x, :] = [255, 255, 255]
            
        cv2.imwrite("debug_rrt.png", frame)