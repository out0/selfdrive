from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from vision.occupancy_grid_cuda import GridDirection
import math, numpy as np
from planner.local_planner.executors.dubins_curves import Dubins
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from model.physical_parameters import PhysicalParameters
from .debug_dump import dump_result
from planner.physical_model import ModelCurveGenerator
from data.coordinate_converter import CoordinateConverter
from model.map_pose import MapPose

DEBUG_DUMP = False

class OvertakerPlanner(LocalPathPlannerExecutor):
    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _dubins: Dubins
    _kinematics: ModelCurveGenerator
    _coord_conv: CoordinateConverter

    NAME = "overtaker"

    def __init__(self, 
                 max_exec_time_ms: int, 
                 coord_converter: CoordinateConverter) -> None:
        
        super().__init__(max_exec_time_ms)
        self._dubins = Dubins(40, 4)
        self._kinematics = ModelCurveGenerator(0.05)
        self._coord_conv = coord_converter


    def plan(self, planner_data: PlanningData, partial_result: PlanningResult) -> None:
        self._og = planner_data.og
        self._planner_data = planner_data
        self._search = True
        self._result = partial_result
        self._plan_task = Thread(target=self.__perform_local_planning)
        self._plan_task.start()

    def cancel(self) -> None:
        self._search = False
        self._plan_task = None

    def is_planning(self) -> bool:
        return self._search

    def get_result(self) -> PlanningResult:
        return self._result

    def __check_path_feasible(self, path: list[Waypoint]) -> bool:
        if path is None or len(path) <= 2:
            return False
        
        return self._planner_data.og.check_all_path_feasible(path)
    
    def __build_overtake_path(self, start: Waypoint, goal: Waypoint) -> list[Waypoint]:
        
        dx = goal.x - start.x
        location: MapPose = self._planner_data.ego_location      
        map_start: MapPose = self._coord_conv.convert_waypoint_to_map_pose(location, start)
        velocity: float = 2
                
        if dx < 0:
            steering_angle = -PhysicalParameters.MAX_STEERING_ANGLE
        elif dx > 0:
            steering_angle = PhysicalParameters.MAX_STEERING_ANGLE

        lr = self._kinematics.get_lr()

        steer = math.tan(math.radians(steering_angle))
        beta = math.atan(steer / lr)
        dt = abs((dx * PhysicalParameters.OG_HEIGHT_PX_TO_METERS_RATE) / (velocity * math.cos(beta)))
        
        path1 = self._kinematics.gen_path_cg_by_driving_time(map_start, steering_angle, velocity, 2.5*dt, 10)
        #path2 = self._kinematics.gen_path_cg_by_driving_time(path1[-1], -steering_angle, velocity, 0.6*dt, 10)
        #path1.extend(path2)
        path = self._coord_conv.convert_map_path_to_waypoint(location, path1)
        
        path3 = WaypointInterpolator.interpolate_straight_line_path2(
            path[-1], goal,  self._planner_data.og.width(), self._planner_data.og.height(), 20)
        
        path.extend(path3)
        
        return path
  

    def __perform_local_planning(self) -> None:
        self._result.planner_name = OvertakerPlanner.NAME
        self._result.result_type = PlannerResultType.INVALID_PATH
        self.set_exec_started()
        self._rst_timeout()

        start = self._result.local_start
        goal = Waypoint(self._result.local_goal.x,
                        self._result.local_goal.z)


        if self.__try_direct_path(goal):
            if DEBUG_DUMP:
                dump_result(self._og, self._result)
            self._result.result_type = PlannerResultType.VALID
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return
        
        # if fails, try relocating the goal point       
        try_left_first = self._result.local_goal.x < start.x
        
        if try_left_first:
            if self.__relocate_left(goal):
                self._result.result_type = PlannerResultType.VALID
                self._result.total_exec_time_ms = self.get_execution_time()
                self._search = False
                return
            if self.__relocate_right(goal):
                self._result.result_type = PlannerResultType.VALID
                self._result.total_exec_time_ms = self.get_execution_time()
                self._search = False
                return
        else:
            if self.__relocate_right(goal):
                self._result.result_type = PlannerResultType.VALID
                self._result.total_exec_time_ms = self.get_execution_time()
                self._search = False
                return
            if self.__relocate_left(goal):
                self._result.result_type = PlannerResultType.VALID
                self._result.total_exec_time_ms = self.get_execution_time()
                self._search = False
                return
        
        self._result.result_type = PlannerResultType.INVALID_PATH
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False
    
    def __find_first_feasible_goal(self, z: int, x_init: int, x_limit: int) -> int:
        inc = 1
        if x_init > x_limit:
            inc = -1
        
        for i in range(x_init, x_limit, inc):
            if self._og.check_direction_allowed(i, z, GridDirection.HEADING_0):
                return i
        return -1
    
    def __try_direct_path(self, goal: Waypoint):
        start = self._result.local_start
        
        self._result.path = self.__build_overtake_path(start, goal)
        
        if DEBUG_DUMP:
            dump_result(self._og, self._result)
        
        return self.__check_path_feasible(self._result.path)
    
    def __relocate_left(self, goal: Waypoint) -> bool:
        
        x_min = self.__find_first_feasible_goal(goal.z, 0, goal.x)
        
        if x_min < 0:
            return False
        
        x = math.floor(0.5 * (goal.x + x_min))
        
        if x < x_min or x > goal.x:
            return False

        start = self._result.local_start
        new_goal = Waypoint(x, goal.z, 0)
        self._result.path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            dump_result(self._og, self._result)
            
        if self._og.check_all_path_feasible(self._result.path):
            return True
        
        return self.__relocate_step_left(goal, 2, x_min)
    
    def __relocate_right(self, goal: Waypoint) -> bool:
        
        x_max = self.__find_first_feasible_goal(goal.z, goal.x, self._og.width())
        
        if x_max < 0:
            return False
        
        x = math.floor(0.5 * (goal.x + x_max))
        
        if x < goal.x or x > x_max:
            return False

        start = self._result.local_start
        new_goal = Waypoint(x, goal.z, 0)
        self._result.path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            dump_result(self._og, self._result)
            
        if self._og.check_all_path_feasible(self._result.path):
            return True
        
        return self.__relocate_step_right(goal, 2, x_max)
    
    def __relocate_step_left(self, goal: Waypoint, step: int, x_min: int) -> bool:
        
        if x_min < 0:
            return False
        
        x = goal.x - step
        
        if x < x_min:
            return False

        start = self._result.local_start
        new_goal = Waypoint(x, goal.z, 0)
        self._result.path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            dump_result(self._og, self._result)
            
        if self._og.check_all_path_feasible(self._result.path):
            return True
        
        self.__relocate_step_left(new_goal, step, x_min)
            
    def __relocate_step_right(self, goal: Waypoint, step: int, x_max: int) -> bool:

        if x_max < 0:
            return False
        
        x = goal.x + step
        
        if x >= x_max:
            return False

        start = self._result.local_start
        new_goal = Waypoint(x, goal.z, 0)
        self._result.path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            dump_result(self._og, self._result)
            
        if self._og.check_all_path_feasible(self._result.path):
            return True
        
        self.__relocate_step_right(new_goal, step, x_max)        
    
   