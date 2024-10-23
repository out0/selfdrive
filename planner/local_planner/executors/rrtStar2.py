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

DEBUG_DUMP = True

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
        
    NAME = "RRT"
    
    RRT_STEP = 15
    MAX_STEP = 30
    
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._compute_frame = CudaGraph(PhysicalParameters.OG_WIDTH, PhysicalParameters.OG_HEIGHT)
        self._node_list = []
        self._num_nodes_in_list = 0
        self._start = Waypoint(128, PhysicalParameters.EGO_UPPER_BOUND.z - 1)
        self._og = None
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self._search = True
        self._num_nodes_in_list = 0
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


    def __compute_distance(self, x1, z1, x2: int, z2: int) -> float:
        dx = x2 - x1
        dz = z2 - z1
        return dx ** 2 + dz ** 2
        
    def __generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
    
    def __generate_node_towards_point(self, base_point_x: int, base_point_z: int, x_rand: int, z_rand: int) -> tuple[int, int]:
        dist = math.sqrt(self.__compute_distance(base_point_x, base_point_z, x_rand, z_rand))
        
        slope = math.atan2(z_rand - base_point_z, x_rand - base_point_x)
        
        if dist < RRTPlanner.MAX_STEP:            
            x = base_point_x + math.floor(dist * math.cos(slope))
            z = base_point_z - math.floor(dist * math.sin(slope))
            return x, z
        
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point_z, x_rand - base_point_x)
        x = base_point_x + math.floor(RRTPlanner.MAX_STEP * math.cos(slope))
        z = base_point_z + math.floor(RRTPlanner.MAX_STEP * math.sin(slope))
        return x, z
    
    def __check_path_feasible(self, nearest: tuple[int, int], new_p: tuple[int, int]) -> bool:
        path = WaypointInterpolator.interpolate_straight_line_path2(
            Waypoint(nearest[0], nearest[1]), 
            Waypoint(new_p[0], new_p[1]),
            PhysicalParameters.OG_WIDTH,
            PhysicalParameters.OG_HEIGHT,
            30)
        
        return self._og.check_all_path_feasible(path)
    
    def __to_waypoint(self, p: tuple[int, int]) -> Waypoint:
        if p is None:
            return p
        
        return Waypoint(
            p[0],
            p[1]
        )
        
    def dump_result(self, node_list):
        
        frame = self._og.get_color_frame()
        start = self._goal_result.start
        
        
        for n in node_list:
            parent = self._compute_frame.get_parent(n[0], n[1])
            if parent is None:
                x, z = start.x, start.z
            else:
                x, z = parent
                
            path = WaypointInterpolator.interpolate_straight_line_path2(
                Waypoint(x, z), 
                Waypoint(n[0], n[1]),
                PhysicalParameters.OG_WIDTH,
                PhysicalParameters.OG_HEIGHT,
                30)
            
            for p in path:
                frame[p.z, p.x, :] = [255, 255, 255]
            
        cv2.imwrite("debug_rrt.png", frame)
            
    
    def __perform_planning(self) -> None:
        #self._result.planner_name = RRTPlanner.NAME
        self.set_exec_started()
        
        self._compute_frame.add_point(self._start.x, self._start.z, -1, -1, 0)
        
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            if self._check_timeout():
                loop_search = False
                continue
            
            x_rand, z_rand = self.__generate_random_point()
            
            nearest = self._compute_frame.find_best_neighbor(x_rand, z_rand, 999999999)
            
            if nearest is None:
                continue
            
            (new_x, new_z) = self.__generate_node_towards_point(nearest[0], nearest[1], x_rand, z_rand)
            
            if not self.__check_path_feasible(nearest, (new_x, new_z)):
                continue
            
            best = self._compute_frame.find_best_neighbor(new_x, new_z, RRTPlanner.MAX_STEP)
            
            if best is None:
                continue
            
            self._compute_frame.add_point(new_x, new_z, best[0], best[1], self.__compute_distance(new_x, new_z, best[0], best[1]))
            
            
            self._node_list.append((new_x, new_z))
            
            if DEBUG_DUMP:
                self.dump_result(self._node_list)
            
            if self.__compute_distance(new_x, new_z, self._goal_result.goal.x, self._goal_result.goal.z) <= RRTPlanner.RRT_STEP:
                loop_search = False
        
        # finding the path
        current = self.__to_waypoint(self._compute_frame.find_best_neighbor(self._goal_result.goal.x, self._goal_result.goal.z, 999999999))
        
        path: list[Waypoint] = []
        
        while current is not None:
            path.append(current)
            current = self.__to_waypoint(self._compute_frame.get_parent(current.x, current.z))
        
        path.reverse()
        self._result.path = path
        
        if len(path) <= 3:
            self._result.result_type = PlannerResultType.INVALID_PATH
        else:
            self._result.result_type = PlannerResultType.VALID
        
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False
