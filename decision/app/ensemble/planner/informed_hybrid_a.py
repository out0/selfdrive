from . hybrid_a import HybridAStar
from pydriveless import MapPose, Waypoint, CoordinateConverter, PI
from pydriveless import SearchFrame, angle, SearchParams, EgoParams
from pydriveless import Interpolator
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlannerResultType

class InformedHybridAStar(LocalPlannerExecutor):
    __subplanner: HybridAStar
    __sub_goal_list: list[Waypoint]
    __goal_list: list[Waypoint]
    __goal_list_pos: int
    __path: list[Waypoint]
    _ego_params: EgoParams
    _search_params: SearchParams

    def __init__(self, ego_params: EgoParams):
        super().__init__("Informed Hybrid A*")
        self.__subplanner = HybridAStar(ego_params)
        self.__sub_goal_list = []
        self._ego_params = ego_params
        
    def inform_sub_goals(self, subgoals: list[Waypoint]) -> None:
        self.__sub_goal_list = subgoals

    def _planning_init(self, search_params: SearchParams) -> bool:
        self.__goal_list = [search_params.start]
        self.__goal_list.extend(self.__sub_goal_list)
        self.__goal_list.append(search_params.goal)
        self.__goal_list_pos = 1
        self.__path = []
        self._search_params = search_params
        return True

    def _loop_plan(self) -> bool:

        if self.__goal_list_pos >= len(self.__goal_list):
            self._set_planning_result(PlannerResultType.VALID, self.__path)
            return False
        
        start_location = self.__goal_list[self.__goal_list_pos - 1]
        start_location_map = self._ego_params.coordinate_converter().convert(self._search_params.map_origin, start_location)

        sub_search = self._search_params.clone()
        sub_search._start = start_location
        sub_search._ego_pose = start_location_map
        sub_search._goal = self.__goal_list[self.__goal_list_pos]


        self.__subplanner._planning_init(sub_search)
        while self.__subplanner._loop_plan():
            if self._check_timeout():
                return False
        
        res = self.__subplanner.get_result()
        if res is None or res.result_type != PlannerResultType.VALID:
            self._set_planning_result(PlannerResultType.INVALID_PATH, self.__path)
            return False
        
        self.__goal_list_pos += 1
        self.__path.extend(res.path)
        return True

    def _loop_optimize(self) -> bool:
        return False
    