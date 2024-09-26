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
from planner.goal_point_discover import GoalPointDiscoverResult

DEBUG_DUMP = True

class OvertakerPlanner(LocalPathPlannerExecutor):
    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _goal_result: GoalPointDiscoverResult
    _dubins: Dubins
    _kinematics: ModelCurveGenerator
    _coord_conv: CoordinateConverter
    __path: list[Waypoint]

    NAME = "overtaker"

    def __init__(self, 
                 max_exec_time_ms: int, 
                 coord_converter: CoordinateConverter) -> None:
        
        super().__init__(max_exec_time_ms)
        self._dubins = Dubins(40, 4)
        self._kinematics = ModelCurveGenerator(0.05)
        self._coord_conv = coord_converter
        self.__path = None
        self._goal_result = None


    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._og = planner_data.og
        self._planner_data = planner_data
        self._result = None
        self._goal_result = goal_result
        self._search = True
        
        if goal_result.goal is None:
            self._result = PlanningResult.build_basic_response_data(
                OvertakerPlanner.NAME,
                PlannerResultType.INVALID_GOAL,
                planner_data,
                goal_result
            )
            self._search = False
            return
        
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
        
        return self._dubins.build_path(start, goal, self._og.width() - 1, self._og.height() - 1)
  

    def __perform_local_planning(self) -> None:
        self.set_exec_started()
        self._rst_timeout()
        
        local_start = Waypoint(
            128,
            128,
            0
        )
        
        goal = Waypoint(self._goal_result.goal.x,
                        self._goal_result.goal.z,
                        self._goal_result.goal.heading)


        if self.__try_direct_path(goal):                
            self._result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            
            if DEBUG_DUMP:
                dump_result(self._og, self._result)
            
            self._search = False
            return
        
        # if fails, try relocating the goal point       
        try_left_first = self._goal_result.goal.x < local_start.x
        
        valid = False
        
        if try_left_first:
            if self.__relocate_left(goal):
                valid = True
            elif self.__relocate_right(goal):
                valid = True
        else:
            if self.__relocate_right(goal):
                valid = True
            elif self.__relocate_left(goal):
                valid = True
        
        if valid:
            self._result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
        else:
            self._result = PlanningResult.build_basic_response_data(
                OvertakerPlanner.NAME,
                PlannerResultType.INVALID_GOAL,
                self._planner_data,
                self._goal_result,
                total_exec_time_ms = self.get_execution_time()
            )
        self._search = False

    
    def __find_first_feasible_goal(self, z: int, x_init: int, x_limit: int, heading: float) -> int:
        inc = 1
        if x_init > x_limit:
            inc = -1
        
        for i in range(x_init, x_limit, inc):
            if self._og.check_waypoint_feasible(Waypoint(i, z, heading)):
                return i
        return -1
    
    def __try_direct_path(self, goal: Waypoint):
        start = self._goal_result.start
        
        self.__path = self.__build_overtake_path(start, goal)
        
        if DEBUG_DUMP:
            result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            dump_result(self._og, result)
        
        return self.__check_path_feasible(self.__path)
    
    def __relocate_left(self, goal: Waypoint) -> bool:
        
        x_min = self.__find_first_feasible_goal(goal.z, 0, goal.x, goal.heading)
        
        if x_min < 0:
            return False
        
        x = math.floor(0.5 * (goal.x + x_min))
        
        if x < x_min or x > goal.x:
            return False

        start = self._goal_result.start
        new_goal = Waypoint(x, goal.z, goal.heading)
        self.__path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            
            dump_result(self._og, result)
            
        if self._og.check_all_path_feasible(self.__path):
            return True
        
        return self.__relocate_step_left(goal, 2, x_min)
    
    def __relocate_right(self, goal: Waypoint) -> bool:
        
        x_max = self.__find_first_feasible_goal(goal.z, goal.x, self._og.width(), goal.heading)
        
        if x_max < 0:
            return False
        
        x = math.floor(0.5 * (goal.x + x_max))
        
        if x < goal.x or x > x_max:
            return False

        start = self._goal_result.start
        new_goal = Waypoint(x, goal.z, goal.heading)
        self.__path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            
            result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            
            dump_result(self._og, result)
            
        if self._og.check_all_path_feasible(self.__path):
            return True
        
        return self.__relocate_step_right(goal, 2, x_max)
    
    def __relocate_step_left(self, goal: Waypoint, step: int, x_min: int) -> bool:
        
        if x_min < 0:
            return False
        
        x = goal.x - step
        
        if x < x_min:
            return False

        start = self._goal_result.start
        new_goal = Waypoint(x, goal.z, 0)
        self.__path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            
            dump_result(self._og, result)
            
        if self._og.check_all_path_feasible(self.__path):
            return True
        
        self.__relocate_step_left(new_goal, step, x_min)
            
    def __relocate_step_right(self, goal: Waypoint, step: int, x_max: int) -> bool:

        if x_max < 0:
            return False
        
        x = goal.x + step
        
        if x >= x_max:
            return False

        start = self._goal_result.start
        new_goal = Waypoint(x, goal.z, 0)
        self.__path = self.__build_overtake_path(start, new_goal)
        
        if DEBUG_DUMP:
            
            result = PlanningResult(
                planner_name = OvertakerPlanner.NAME,
                ego_location = self._planner_data.ego_location,
                goal = self._planner_data.goal,
                next_goal = self._planner_data.next_goal,
                local_start = self._goal_result.start,
                local_goal = self._goal_result.goal,
                direction = self._goal_result.direction,
                timeout = False,
                path = self.__path,
                result_type = PlannerResultType.VALID,
                total_exec_time_ms = self.get_execution_time()
            )
            
            dump_result(self._og, result)
            
        if self._og.check_all_path_feasible(self.__path):
            return True
        
        self.__relocate_step_right(new_goal, step, x_max)        
    
   