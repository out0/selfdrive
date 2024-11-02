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
from planner.local_planner.executors.rrtStar2 import RRTPlanner
# from planner.local_planner.executors.dubins_path import DubinsPathPlanner
from planner.goal_point_discover import GoalPointDiscoverResult
import threading

class HierarchicalGroupPlanner(LocalPathPlannerExecutor):
    __interpolator: InterpolatorPlanner
    __overtaker: OvertakerPlanner
    __hybrid__astar: HybridAStarPlanner
    __rrt_star: RRTPlanner
    # _dubins: DubinsPathPlanner
    __max_exec_time_ms: int
    __planner_data: PlanningData
    __exec_plan: bool
    __plan_result: PlanningResult
    __goal_result: GoalPointDiscoverResult
    __plan_thr: Thread


    def __init__(self, 
                map_converter: CoordinateConverter, 
                max_exec_time_ms: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self.__interpolator = InterpolatorPlanner(map_converter, max_exec_time_ms)
        self.__overtaker = OvertakerPlanner(max_exec_time_ms, map_converter)
        self.__hybrid__astar = HybridAStarPlanner(max_exec_time_ms, map_converter, 10)
        self.__rrt_star = RRTPlanner(max_exec_time_ms, 50)
        #self.__astar = VectorialAStarPlanner(max_exec_time_ms)
        # self._dubins = DubinsPathPlanner(max_exec_time_ms, map_converter, )
        self.__exec_plan = False
        self.__max_exec_time_ms = max_exec_time_ms
        self.__planner_data = None
        self.__plan_result = None
        self.__goal_result = None

    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        
        self.__planner_data = planner_data
        self.__exec_plan = True
        self.__plan_result = None
        self.__goal_result = goal_result
        
        self.__plan_thr = Thread(target=self.__execute_supervised_planning)
        self.__plan_thr.start()


    def cancel(self) -> None:
        self.__exec_plan = False
        self.__interpolator.cancel()
        self.__hybrid__astar.cancel()
        self.__overtaker.cancel()
        self.__rrt_star.cancel()

    def is_planning(self) -> bool:
        return self.__exec_plan
        
    def get_result(self) -> PlanningResult:
        res = self.__plan_result
        return res
    
    def __timeout(self, start_time: int, max_exec_time_ms: int):
        return 1000*(time.time() - start_time) > max_exec_time_ms
    
    def __on_full_cancelled_planning(self) -> PlanningResult:
        self.__exec_plan = False

    def __wait_execution(self, start_time: int, method: callable) -> bool:
        while self.__exec_plan and method():
            if self.__max_exec_time_ms > 0 and self.__timeout(start_time, self.__max_exec_time_ms):
                self.__on_full_cancelled_planning()
                return True
            time.sleep(0.001)
        return False
    

    def __check_planner(self, start_time, planner: LocalPathPlannerExecutor ) -> bool:
        
        timeout = self.__wait_execution(start_time, lambda: planner.is_planning())
        
        if timeout:
            self.__exec_plan = False
            return True
        
        self.__plan_result = planner.get_result()
        
        if self.__plan_result is None:
            is_valid = False
        else:
            is_valid = self.__plan_result.result_type == PlannerResultType.VALID
        
        if is_valid:
            self.cancel()
        
        return is_valid
    
    def __execute_supervised_planning(self) -> None:
        self.__exec_plan = True
        self.__interpolator.plan(self.__planner_data, self.__goal_result)
        #self.__astar.plan(self.__planner_data, self.__goal_result)
        self.__hybrid__astar.plan(self.__planner_data, self.__goal_result)
        # self._dubins.plan(self.__planner_data, self.__goal_result)
        self.__overtaker.plan(self.__planner_data, self.__goal_result)
        self.__rrt_star.plan(self.__planner_data, self.__goal_result)

        start_time = time.time()

        check = True

        if self.__check_planner(start_time, self.__interpolator):
            check = False
        
        if check and self.__check_planner(start_time, self.__overtaker):
            check = False
        
        if check and self.__check_planner(start_time, self.__hybrid__astar):
            check = False
        
        if check and self.__check_planner(start_time, self.__rrt_star):
            check = False
            
        # if check:
        #     self.__check_planner(start_time, self.__astar)

        self.__exec_plan = False

