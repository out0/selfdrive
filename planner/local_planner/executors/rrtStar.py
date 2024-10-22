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
    _node_list: np.ndarray
    _num_nodes_in_list: int
        
    NAME = "RRT"
    
    POS_X = 0
    POS_Z = 1
    PARENT_ID = 2
    COST = 3
    
    def __init__(self,  
                 max_exec_time_ms: int, 
                 max_steps: float
                 ) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._compute_frame = CudaGraph(PhysicalParameters.OG_WIDTH, PhysicalParameters.OG_HEIGHT)
        self._max_step = max_steps
        self._node_list = np.zeros((PhysicalParameters.OG_HEIGHT * PhysicalParameters.OG_WIDTH, 4))
        self._num_nodes_in_list = 0
        self._start = Waypoint(128, PhysicalParameters.EGO_UPPER_BOUND.z - 1)
        self._og = None
        
    def plan(self, planner_data: PlanningData, result) -> None:
        
        self._og = planner_data.og
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self._search = True
        self._num_nodes_in_list = 0
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
        
        if dist < self._max_step:            
            x = base_point_x + math.floor(dist * math.cos(slope))
            z = base_point_z - math.floor(dist * math.sin(slope))
            return x, z
        
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point_z, x_rand - base_point_x)
        x = base_point_x + math.floor(self._max_step * math.cos(slope))
        z = base_point_z + math.floor(self._max_step * math.sin(slope))
        return x, z
    
    def __check_path_feasible(self, nearest: tuple[int, int], new_p: tuple[int, int]) -> bool:
        path = WaypointInterpolator.interpolate_straight_line_path2(
            Waypoint(nearest[0], nearest[1]), 
            Waypoint(new_p[0], new_p[1]),
            PhysicalParameters.OG_WIDTH,
            PhysicalParameters.OG_HEIGHT,
            30)
        
        return self._og.check_all_path_feasible(path)
    
    def __add_node(self, x: int, z: int, parent: int, cost: int) -> None:
        self._node_list[self._num_nodes_in_list, RRTPlanner.POS_X] = x
        self._node_list[self._num_nodes_in_list, RRTPlanner.POS_Z] = z
        self._node_list[self._num_nodes_in_list, RRTPlanner.PARENT_ID] = parent
        self._node_list[self._num_nodes_in_list, RRTPlanner.COST] = cost
        self._num_nodes_in_list += 1
    
    def __to_waypoint(self, p: tuple[int, int]) -> Waypoint:
        if p is None:
            return p
        
        return Waypoint(
            p[0],
            p[1]
        )
    
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
            
            best = self._compute_frame.find_best_neighbor(new_x, new_z, self._max_step)
            
            if best is None:
                continue
            
            self._compute_frame.add_point(new_x, new_z, best[0], best[1], self.__compute_distance(new_x, new_z, best[0], best[1]))
            
        
        # finding the path
        current = self.__to_waypoint(self._compute_frame.find_best_neighbor(self._result.local_goal.x, self._result.local_goal.z, 999999999))
        
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
