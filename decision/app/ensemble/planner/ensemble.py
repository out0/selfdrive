
from pydriveless import Waypoint, angle, MapPose
from pydriveless import CoordinateConverter
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
from .interpolator import Interpolator
from .overtaker import Overtaker
from .hybrid_a import HybridAStar
from .bi_rrt import BiRRTStar
from ..model.curve_quality import CurveAssessment, CurveData
import math, time
from threading import Lock

MAX_VALUE = 99999999
PATH_CHANGE_VIABLE_MAX_DIST_PX = 10

class Ensemble(LocalPlannerExecutor):
    _map_coordinate_converter: CoordinateConverter
    _planning_set: list[LocalPlannerExecutor]
    _planning_set_exec_coarse: list[bool]
    _planning_set_exec_optim: list[bool]
    __new_path_available: bool
    _mtx_path: Lock
    _chosen_planner_name: str

    def __init__(self, map_coordinate_converter: CoordinateConverter,
                 max_exec_time_ms: int):
        super().__init__("Ensemble", max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self.__initialize_planners()
        self.__new_path_available = False
        self._mtx_path = Lock()
        self._chosen_planner_name = None
   
    def __initialize_planners(self) -> None:
        planner_interpolator = Interpolator(self._map_coordinate_converter, max_exec_time_ms=self.get_max_exec_time_ms())
        
        planner_overtaker = Overtaker(max_exec_time_ms=self.get_max_exec_time_ms())
        
        planner_hybrid = HybridAStar(
            self._map_coordinate_converter, 
            max_exec_time_ms=self.get_max_exec_time_ms())
        
        planner_bi_rrt = BiRRTStar(map_coordinate_converter=self._map_coordinate_converter, 
                            max_exec_time_ms=self.get_max_exec_time_ms(),
                            max_path_size_px=30,
                            dist_to_goal_tolerance_px=20,
                            class_cost=PhysicalParameters.SEGMENTATION_CLASS_COST)
        
        self._planning_set = [
            planner_interpolator,
            planner_overtaker,
            planner_hybrid,
            planner_bi_rrt
        ]
        
        

    def __terminate_local_planners(self) -> None:
        for p in self._planning_set:
            p.cancel()

    def __assert_local_planners_termination(self) -> None:
        v = True
        while v:
            v = False
            for p in self._planning_set:
                v = v or p.is_running()            
            
            if v: time.sleep(0.01)

    def __check_coarse_planning(self) -> bool:
        for p in self._planning_set_exec_coarse:
            if p: return True
        return False
    
    def __check_optim_planning(self) -> bool:
        for p in self._planning_set_exec_optim:
            if p: return True
        return False    
        
    def cancel(self) -> None:
        super().cancel()
        self.__terminate_local_planners()
        self.__assert_local_planners_termination()
        
    def _planning_init(self, planning_data: PlanningData) -> bool:
        self.__terminate_local_planners()
        self.__assert_local_planners_termination()
        
        planning_data.og().process_safe_distance_zone(planning_data.min_distance(), True)
        planning_data.og().process_distance_to_goal(planning_data.local_goal().x, planning_data.local_goal().z)

        self._planning_set_exec_coarse = [True for _ in self._planning_set]
        self._planning_set_exec_optim = [True for _ in self._planning_set]
        self.__new_path_available = False
        self._best_cost = MAX_VALUE
        self.__last_ego_location = planning_data.ego_location()

        for p in self._planning_set:
            p.plan(planning_data, run_in_main_thread=False)
        return True


    def _loop_plan(self, planning_data: PlanningData) -> bool:
        if not self.__check_coarse_planning():
            return False
        
        for i in range(len(self._planning_set)):
            p = self._planning_set[i]

            if self._planning_set_exec_coarse[i] and not p.is_planning():
                # A new plan is available
                self._planning_set_exec_coarse[i] = False
                result = p.get_result()
                
                if result.result_type == PlannerResultType.VALID and len(result.path) > 0:
                    cost, path_metrics = self.__assess_curve_quality_cost(result.path, planning_data.local_goal())                    
                    if cost < self._best_cost and self.__assess_path_change_viability(planning_data.base_map_conversion_location, result.path):
                        self._mtx_path.acquire()
                        self._best_cost = cost
                        result.planner_name = f"{self.get_planner_name()}: {p.get_planner_name()}"
                        result.result_type = PlannerResultType.VALID
                        result.curve_cost = cost
                        result.path_metrics = path_metrics
                        self._planning_result = result
                        self.__new_path_available = True
                        self._chosen_planner_name = result.planner_name
                        self._mtx_path.release()

        return True

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        if not self.__check_optim_planning():
            return False
        
        for i in range(len(self._planning_set)):
            p = self._planning_set[i]

            if self._planning_set_exec_optim[i] and not p.is_planning():
                # A new plan is available
                self._planning_set_exec_optim[i] = False
                result = p.get_result()
                
                if result.result_type == PlannerResultType.VALID and len(result.path) > 0:
                    cost, path_metrics = self.__assess_curve_quality_cost(result.path, planning_data.local_goal())
                    if cost < self._best_cost and self.__assess_path_change_viability(planning_data.ego_location(), result.path):
                        self._mtx_path.acquire()
                        self._best_cost = cost
                        result.planner_name = f"{self.get_planner_name()}: {p.get_planner_name()}"
                        result.result_type = PlannerResultType.VALID
                        result.curve_cost = cost
                        result.path_metrics = path_metrics
                        self._planning_result = result
                        self.__new_path_available = True
                        self._chosen_planner_name = result.planner_name
                        self._mtx_path.release()
        return True

    def update_last_ego_location(self, val: MapPose) -> None:
        self.__last_ego_location = val

    def new_path_available(self) -> bool:
        return self.__new_path_available
    
    def get_result(self) -> PlanningResult:
        self._mtx_path.acquire(blocking=True)
        res = self._planning_result
        self.__new_path_available = False
        self._mtx_path.release()
        res.planner_name = f"{self.get_planner_name()}: {self._chosen_planner_name}"
        return res
    
    K1 = 0.5
    K2 = 0.5
    K3 = 2
    K4 = 0.5

    def __assess_curve_quality_cost(self, path: list[Waypoint], goal: Waypoint) -> float:
        curve_data: CurveData = CurveAssessment.assess_curve(curve=path)
        heading_error = abs((goal.heading - curve_data.last_p.heading).rad())
        dist_error = Waypoint.distance_between(curve_data.last_p, goal)
        cost =  Ensemble.K1 * curve_data.jerk + Ensemble.K2 * curve_data.total_length +\
            Ensemble.K3 * heading_error + Ensemble.K4 * dist_error
        if curve_data.tan_discontinuity:
            return 2*cost, curve_data
        return cost, curve_data
    

    def __assess_path_change_viability(self, map_base_pose: MapPose, path: list[Waypoint]) -> bool:
        curr_pos = self._map_coordinate_converter.convert(map_base_pose, self.__last_ego_location)
        for p in path:
            d = Waypoint.distance_between(curr_pos, p)
            if d <= PATH_CHANGE_VIABLE_MAX_DIST_PX:
                return True
