from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
import time

from planner.local_planner.executors.vectorial_astar import VectorialAStarPlanner
from planner.local_planner.executors.interpolator import InterpolatorPlanner
from planner.local_planner.executors.overtaker import OvertakerPlanner
from planner.local_planner.executors.hybridAStar import HybridAStarPlanner
from planner.local_planner.executors.dubins_path import DubinsPathPlanner


class HierarchicalGroupPlanner(LocalPathPlannerExecutor):
    _interpolator: InterpolatorPlanner
    _overtaker: OvertakerPlanner
    _hybrid_astar: HybridAStarPlanner
    _astar: VectorialAStarPlanner
    _dubins: DubinsPathPlanner
    _max_exec_time_ms: int
    _planner_data: PlanningData
    _exec_plan: bool
    _plan_result: PlanningResult
    _plan_thr: Thread


    def __init__(self, 
                map_converter: CoordinateConverter, 
                max_exec_time_ms: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self._interpolator = InterpolatorPlanner(map_converter, max_exec_time_ms)
        #self._overtaker = OvertakerPlanner(max_exec_time_ms, 10)
        self._hybrid_astar = HybridAStarPlanner(max_exec_time_ms, map_converter, 10)
        self._astar = VectorialAStarPlanner(max_exec_time_ms)
        self._dubins = DubinsPathPlanner(max_exec_time_ms, map_converter, )
        self._exec_plan = False
        self._max_exec_time_ms = max_exec_time_ms
        self._planner_data = None
        self._og = None
        self._plan_result = None

    def plan(self, planner_data: PlanningData, partial_result: PlanningResult) -> None:
        
        self._planner_data = planner_data
        self._exec_plan = True
        self._plan_result = partial_result
        self._plan_thr = Thread(target=self.__execute_supervised_planning)
        self._plan_thr.start()


    def cancel(self) -> None:
        self._exec_plan = False
        self._interpolator.cancel()
        self._astar.cancel()
        self._hybrid_astar.cancel()
        #self._overtaker.cancel()
        self._plan_result = None
        



    def is_planning(self) -> bool:
        return self._exec_plan
        

    def get_result(self) -> PlanningResult:
        return self._plan_result
    
    def __timeout(self, start_time: int, max_exec_time_ms: int):
        return 1000*(time.time() - start_time) > max_exec_time_ms
    
    def __on_full_cancelled_planning(self) -> PlanningResult:
        self._plan_result.timeout = True
        self._exec_plan = False

    def __got_timeout(self, start_time: int, method: callable) -> bool:
        while self._exec_plan and method():
            if self._max_exec_time_ms > 0 and self.__timeout(start_time, self._max_exec_time_ms):
                self.__on_full_cancelled_planning()
                return True
            time.sleep(0.001)
        return False
        
    def set_bounds(self, lower_bound: Waypoint, upper_bound: Waypoint):
        self._interpolator.set_bounds(lower_bound, upper_bound)
        self._astar.set_bounds(lower_bound, upper_bound)
        self._hybrid_astar.set_bounds(lower_bound, upper_bound)
        #self._overtaker.set_bounds(lower_bound, upper_bound)
        pass    
    
    def __execute_supervised_planning(self) -> None:
        self._exec_plan = True
        self._interpolator.plan(self._og, self._planner_data)
        self._astar.plan(self._og,  self._planner_data)
        self._hybrid_astar.plan(self._og,  self._planner_data)
        self._dubins.plan(self._og,  self._planner_data)
        #self._overtaker.plan(self._og,  self._planner_data)

        start_time = time.time()

        if not self.__got_timeout(start_time, lambda: self._interpolator.is_planning()):
            self._plan_result = self._interpolator.get_result()
        
            if self._plan_result.result_type == PlannerResultType.VALID:
                self._astar.cancel()
                self._hybrid_astar.cancel()
                self._dubins.cancel()
                #self._overtaker.cancel()
                self._exec_plan = False
                return
            
            
        if not self.__got_timeout(start_time, lambda: self._dubins.is_planning()):
            self._plan_result = self._dubins.get_result()
        
            if self._plan_result.result_type == PlannerResultType.VALID:
                self._astar.cancel()
                self._hybrid_astar.cancel()
                self._exec_plan = False
                return
    
        if not self.__got_timeout(start_time, lambda: self._hybrid_astar.is_planning()):
            self._plan_result = self._hybrid_astar.get_result()
            if self._plan_result.result_type == PlannerResultType.VALID:
                self._astar.cancel()
                #self._overtaker.cancel()
                self._exec_plan = False
                return
    
        # if not self.__got_timeout(start_time, lambda: self._overtaker.is_planning()):
        #     self._plan_result = self._overtaker.get_result()
        
        #     if self._plan_result.result_type == PlannerResultType.VALID:
        #         self._astar.cancel()
        #         self._exec_plan = False
        #         return
    
        if not self.__got_timeout(start_time, lambda: self._astar.is_planning()):
            self._plan_result = self._astar.get_result()
        
            if self._plan_result.result_type == PlannerResultType.VALID:
                smooth_path = WaypointInterpolator.path_smooth(self._plan_result.path)
        
                if len(smooth_path) < 10:
                    self._exec_plan = False
                    return
                
                if self._planner_data.og.check_path_feasible(smooth_path):
                    self._plan_result.planner_name = f"{self._plan_result.planner_name}+Smooth"
                    self._plan_result.path = smooth_path

        self._exec_plan = False



