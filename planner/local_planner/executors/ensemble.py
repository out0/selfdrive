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
from planner.local_planner.executors.rrtStar import RRTPlanner
# from planner.local_planner.executors.dubins_path import DubinsPathPlanner
from planner.goal_point_discover import GoalPointDiscoverResult
from utils.jerk import Jerk2D

class EnsemblePlanner(LocalPathPlannerExecutor):
    __interpolator: InterpolatorPlanner
    __overtaker: OvertakerPlanner
    __hybrid__astar: HybridAStarPlanner
    __rrt_star: RRTPlanner
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
    
    def __timeout(self, start_time: int, max_exec_time_ms: int) -> bool:
        if max_exec_time_ms < 0: return False
        return 1000*(time.time() - start_time) > max_exec_time_ms
    

    def __check_all_finished(list_finished: list[bool]) -> bool:
        for p in list_finished:
            if not p: return False
        return True
    
    def __select_best(result1: PlanningResult, result2: PlanningResult, vel: float) -> PlanningResult:
        if result1 is None and result2 is not None:
            return result2
        if result2 is None and result1 is not None:
            return result1
        if result1 is None and result2 is None:
            return None
        
        if result1.result_type == PlannerResultType.VALID and result2.result_type != PlannerResultType.VALID:
            return result1
        if result2.result_type == PlannerResultType.VALID and result1.result_type != PlannerResultType.VALID:
            return result2
        # none is valid, return the first one
        if result1.result_type != PlannerResultType.VALID and result1.result_type != PlannerResultType.VALID:
            return result1

        if result1.timeout and not result2.timeout:
            return result2
        if not result1.timeout and result2.timeout:
            return result1

        if len(result1.path) == 0 and len(result2.path) > 0:
            return result2
        if len(result2.path) == 0 and len(result1.path) > 0:
            return result1
        if len(result1.path) == 0 and len(result2.path) == 0:
            return result1

        jerk1 = Jerk2D.compute_path_jerk(result1.path, vel)
        
        jerk2 = Jerk2D.compute_path_jerk(result2.path, vel)
        
        if jerk1 <= jerk2:
            return result1
        
        return result2
        
    
    def __execute_supervised_planning(self) -> None:
        self.__exec_plan = True
        self.__interpolator.plan(self.__planner_data, self.__goal_result)
        self.__hybrid__astar.plan(self.__planner_data, self.__goal_result)
        self.__overtaker.plan(self.__planner_data, self.__goal_result)
        self.__rrt_star.plan(self.__planner_data, self.__goal_result)

        start_time = time.time()

        
        self.__plan_result = None
        plan_finish = [False, False, False, False]
        
        search = True
        

        while search and not self.__timeout(start_time, self.__max_exec_time_ms):
            if EnsemblePlanner.__check_all_finished(plan_finish):
                search = False
                continue
            
            if not plan_finish[0] and not self.__interpolator.is_planning():
                self.__plan_result = EnsemblePlanner.__select_best(self.__plan_result, self.__interpolator.get_result(), self.__planner_data.velocity)
                plan_finish[0] = True

            if not plan_finish[1] and not self.__overtaker.is_planning():
                self.__plan_result = EnsemblePlanner.__select_best(self.__plan_result, self.__overtaker.get_result(), self.__planner_data.velocity)
                plan_finish[1] = True

            if not plan_finish[2] and not self.__hybrid__astar.is_planning():
                self.__plan_result = EnsemblePlanner.__select_best(self.__plan_result, self.__hybrid__astar.get_result(), self.__planner_data.velocity)
                plan_finish[2] = True

            if not plan_finish[3] and not self.__rrt_star.is_planning():
                self.__plan_result = EnsemblePlanner.__select_best(self.__plan_result, self.__rrt_star.get_result(), self.__planner_data.velocity)
                plan_finish[3] = True

        self.cancel()
        
        # r1 = self.__interpolator.get_result()
        # r2 = self.__overtaker.get_result()
        # r3 = self.__hybrid__astar.get_result()
        # r4 = self.__rrt_star.get_result()
        # p1 = 2

