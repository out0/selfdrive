from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.physical_model import ModelCurveGenerator
from planner.local_planner.executors.dubins_curves import Dubins
from planner.physical_model import ModelCurveGenerator

class DubinsPathPlanner(LocalPathPlannerExecutor):
    _plan_task: Thread
    _search: bool
    _og: OccupancyGrid
    _planner_data: PlanningData
    _result: PlanningResult
    _map_converter: CoordinateConverter

    NAME = "Dubins Path"

    def __init__(self, 
                 max_exec_time_ms: int,
                 map_converter: CoordinateConverter) -> None:
        
        super().__init__(max_exec_time_ms)
        self._map_converter = map_converter


    def plan(self,
             planner_data: PlanningData,
             partial_result: PlanningResult) -> None:
        self._planner_data = planner_data
        self._result = partial_result
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
        self.set_exec_started()

        radius = ModelCurveGenerator.get_min_radius()

        if self._result.map_next_goal is not None:
            p2 = self._map_converter.convert_map_to_waypoint(self._result.ego_location, self._result.map_next_goal)
            self._result.local_goal.heading =  OccupancyGrid.compute_heading(self._result.local_goal, p2)
            

        res = Dubins(5, 1).dubins_path(
            (self._result.local_start.x, self._result.local_start.z, self._result.local_start.heading),
            (self._result.local_goal.x, -90 + self._result.local_goal.z, self._result.local_goal.heading)
        )
        
        self._result.path = []
        
        for p in res:
            if p[0] < 0 or p[0] > self._og.width():
                continue
            if p[1] < 0 or p[1] > self._og.height():
                continue
            self._result.path.append(Waypoint(p[0], p[1], 0))
        
        self._result.result_type = PlannerResultType.INVALID_PATH
        if self._og.check_all_path_feasible(self._result.path):
            self._result.result_type = PlannerResultType.VALID
        
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False


