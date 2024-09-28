
from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from scipy.ndimage import gaussian_filter
from .debug_dump import dump_result
from planner.goal_point_discover import GoalPointDiscoverResult

DEBUG_DUMP = True

class InterpolatorPlanner(LocalPathPlannerExecutor):
    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _map_coordinate_converter: CoordinateConverter
    _goal_result: GoalPointDiscoverResult

    NAME = "interpolator"

    def __init__(self, 
                 map_coordinate_converter: CoordinateConverter, 
                 max_exec_time_ms: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self._result = None
        self._planner_data = None
        self._search = False


    def plan(self, planning_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._planner_data = planning_data
        self._result  = None
        self._search = True
        self._goal_result = goal_result
        
        if goal_result.goal is None:
            self._result = PlanningResult.build_basic_response_data(
                InterpolatorPlanner.NAME,
                PlannerResultType.INVALID_GOAL,
                planning_data,
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


    def __perform_local_planning(self) -> None:
        self.set_exec_started()
        
        next_goal = None
        if self._planner_data.next_goal is not None:
            next_goal = self._map_coordinate_converter.convert_map_to_waypoint(self._planner_data.ego_location, self._planner_data.next_goal)
        
        path = WaypointInterpolator.path_interpolate(self._goal_result.start, self._goal_result.goal, next_goal, self._planner_data.og.height())

        if path is None:
            self._result = PlanningResult.build_basic_response_data(
                InterpolatorPlanner.NAME,
                PlannerResultType.INVALID_PATH,
                self._planner_data,
                self._goal_result,
                total_exec_time_ms=self.get_execution_time()
            )            
            self._search = False
               
        dedup = set()
        new_path = []
        for p in path:
            k = 256 * p.z + p.x
            if k in dedup: continue
            dedup.add(k)
            new_path.append(p)
            
        
        result_type = PlannerResultType.INVALID_PATH
        
        if self._planner_data.og.check_all_path_feasible(new_path):
            result_type = PlannerResultType.VALID

       
        self._result = PlanningResult(
            planner_name = InterpolatorPlanner.NAME,
            ego_location = self._planner_data.ego_location,
            goal = self._planner_data.goal,
            next_goal = self._planner_data.next_goal,
            local_start = self._goal_result.start,
            local_goal = self._goal_result.goal,
            direction = self._goal_result.direction,
            timeout = False,
            path = new_path,
            result_type = result_type,
            total_exec_time_ms = self.get_execution_time()
        )
        self._search = False
        
        if DEBUG_DUMP:
            dump_result(self._planner_data.og, self._result)
        

