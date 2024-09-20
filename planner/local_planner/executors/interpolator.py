
from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from scipy.ndimage import gaussian_filter
from .debug_dump import dump_result

DEBUG_DUMP = False

class InterpolatorPlanner(LocalPathPlannerExecutor):
    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _map_coordinate_converter: CoordinateConverter

    NAME = "interpolator"

    def __init__(self, 
                 map_coordinate_converter: CoordinateConverter, 
                 max_exec_time_ms: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self._result = None
        self._planner_data = None
        self._search = False


    def plan(self, planning_data: PlanningData, partial_result: PlanningResult) -> None:
        self._planner_data = planning_data
        self._result  = partial_result.clone()
        self._result.planner_name = InterpolatorPlanner.NAME
        self._search = True
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
        self._result.planner_name = InterpolatorPlanner.NAME
        self.set_exec_started()
        path: list[Waypoint] = None
        
        path = [
            self._result.local_start,
            self._result.local_goal
        ]

        if self._result.map_next_goal is not None:
            next_possible_local_goal = self._map_coordinate_converter.convert_map_to_waypoint(
                location=self._result.ego_location,
                target=self._result.map_next_goal
            )
            path.append(next_possible_local_goal)
            self._result.path = WaypointInterpolator.path_interpolate(path, next_possible_local_goal, self._planner_data.og.height())
        else:
            self._result.path = WaypointInterpolator.path_interpolate(path, self._result.local_goal, self._planner_data.og.height())

        dedup = set()
        new_path = []
        for p in self._result.path:
            k = 256 * p.z + p.x
            if k in dedup: continue
            dedup.add(k)
            new_path.append(p)
            
        self._result.path = new_path

        if self._result.path is None:
            self._result.result_type = PlannerResultType.INVALID_PATH
            self._result.total_exec_time_ms = self.get_execution_time()
            self._search = False
            return
        
        self._result.result_type = PlannerResultType.INVALID_PATH
        
        if self._planner_data.og.check_all_path_feasible(self._result.path):
            self._result.result_type = PlannerResultType.VALID
        # else:
        #     self.dump_log("interpolator", self._planner_data.og, self._result.path)
       
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False
        
        if DEBUG_DUMP:
            dump_result(self._planner_data.og, self._result)
        

