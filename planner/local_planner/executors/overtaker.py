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

class OvertakerPlanner(LocalPathPlannerExecutor):

    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _step_size: int
    _mid_x: int
    _mid_z: int
    _dubins: Dubins

    NAME = "overtaker"

    def __init__(self, 
                 max_exec_time_ms: int, 
                 step_size: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self._step_size = step_size
        self._dubins = Dubins(40, 4)


    def plan(self, planner_data: PlanningData, partial_result: PlanningResult) -> None:
        self._og = planner_data.og
        self._mid_x = math.floor(planner_data.og.width() / 2)
        self._mid_z = math.floor(planner_data.og.height() / 2)
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
    
    # def __dump_to_tmp(frame: np.ndarray, path: List[Waypoint], color = [255, 255, 255]):
    #     new_f = np.zeros(frame.shape)
    #     for i in range (frame.shape[0]):
    #         for j in range (frame.shape[1]):
    #             new_f[i, j, 0] = frame[i, j, 0]
    #             new_f[i, j, 1] = frame[i, j, 1]
    #             new_f[i, j, 2] = frame[i, j, 2]
                
    #     for p in path:
    #         frame[p.z, p.x, :] = color
    #     cv2.imwrite("overtake_tmp.png", frame)


    def __perform_local_planning(self) -> None:
        self.set_exec_started()
        self._rst_timeout()

        start = self._result.local_start
        goal = Waypoint(self._result.local_goal.x,
                        self._result.local_goal.z)


        if start.x == goal.x:
            # try going straight first        
            path = WaypointInterpolator.interpolate_straight_line_path2(start, goal,  self._og.width, self._og.height, 20)
        else:
            path = self._dubins.build_path(start, goal, self._og.width(), self._og.height())
        
        
        if self.__check_path_feasible(path):
            self._result.result_type = PlannerResultType.VALID
            self._result.path = path
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return
        
        # if fails, try relocating the goal point
        
        try_left_first = self._result.local_goal.x > start.x
        
        if try_left_first:
            self._relocate(start, goal, 0, PhysicalParameters.OG_WIDTH - 1)
        else:
            self._relocate(start, goal, PhysicalParameters.OG_WIDTH - 1, 0)
    
    def __find_first_feasible_goal(self, z: int, x_init: int, x_limit: int) -> int:
        inc = 1
        if x_init > x_limit:
            inc = -1
        
        for i in range(x_init, x_limit, inc):
            if self._og.check_direction_allowed(i, z, GridDirection.HEADING_0):
                return i
        return -1
    
    
    
    def _relocate(self, start: Waypoint, goal: Waypoint, x_min: int, x_max: int):
        x = self.__find_first_feasible_goal(goal.z, x_min, x_max)
        
        if x == -1:
            self._result.result_type = PlannerResultType.INVALID_PATH
            self._result.path = None
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return
       
        self._result.path = self._dubins.build_path(start, Waypoint(x, goal.z, 0), self._og.width(), self._og.height())
        self._result.result_type = PlannerResultType.INVALID_PATH
            
        if self._og.check_all_path_feasible(self._result.path):
            self._result.result_type = PlannerResultType.VALID
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
        else:
            if x_max > x_min:
                self._relocate(start, goal, x + 1, x_max)
            else:
                self._relocate(start, goal, x - 1, x_max)