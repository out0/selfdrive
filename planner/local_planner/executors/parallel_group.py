from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
import time
from planner.local_planner.executors.interpolator import InterpolatorPlanner
from planner.local_planner.executors.overtaker import OvertakerPlanner
from planner.local_planner.executors.hybridAStar import HybridAStarPlanner
from planner.local_planner.executors.rrtStar2 import RRTPlanner
from planner.goal_point_discover import GoalPointDiscoverResult
from utils.jerk import CurveAssessment, CurveData

    

class ParallelGroupPlanner(LocalPathPlannerExecutor):
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
    __path_version: int


    def __init__(self, 
                map_converter: CoordinateConverter, 
                max_exec_time_ms: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self.__interpolator = InterpolatorPlanner(map_converter, max_exec_time_ms)
        self.__overtaker = OvertakerPlanner(max_exec_time_ms, map_converter)
        self.__hybrid__astar = HybridAStarPlanner(max_exec_time_ms, map_converter, 10)
        self.__rrt_star = RRTPlanner(max_exec_time_ms, 500)
        self.__exec_plan = False
        self.__max_exec_time_ms = max_exec_time_ms
        self.__planner_data = None
        self.__plan_result = None
        self.__goal_result = None
        self.__path_version = 0

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
    
    def __check_someone_is_planning(self, planning_set: list[bool]) -> bool:
        for p in planning_set:
            if p: return True
        return False
    
    def __check_planner_coarse(self, planner: LocalPathPlannerExecutor) -> tuple[bool, bool, PlanningResult, CurveData]:

        if planner.NAME == "Hybrid A*":
            pass
        
        if planner.is_planning():
            return [False, True, None, None]
        
        if planner.NAME == "Hybrid A*":
            pass
        
        res = planner.get_result()
        if res == None:
            return [False, False, None, None]
        
        if res.result_type == PlannerResultType.TOO_CLOSE:
            return [True, False, None, None]
        
        elif res.result_type == PlannerResultType.VALID:
            data = CurveAssessment.assess_curve(res.path, res.local_start.heading)
            return [False, False, res, data]
        
        else:
            return [False, False, None, None]
    
    def __check_planner_optim(self, planner: LocalPathPlannerExecutor) -> tuple[bool, bool, PlanningResult, CurveData]:
        if planner.is_optimizing():
            return [False, True, None, None]
                
        res = planner.get_result()
        if res == None:
            return [False, False, None, None]
        
        if res.result_type == PlannerResultType.VALID:
            data = CurveAssessment.assess_curve(res.path, res.local_start.heading)
            return [False, False, res, data]
        
        else:
            return [False, False, None, None]
        
    K1 = 0.5
    K2 = 0.5
    K3 = 2
    K4 = 0.5
    def __curve_quality_cost(self, q: CurveData, goal: Waypoint) -> float:
        heading_error = abs(goal.heading - q.last_p.heading)
        dist_error = Waypoint.distance_between(q.last_p, goal)
        cost =  ParallelGroupPlanner.K1 * q.jerk + ParallelGroupPlanner.K2 * q.total_length +\
            ParallelGroupPlanner.K3 * heading_error + ParallelGroupPlanner.K4 * dist_error
        if q.tan_discontinuity:
            return 2*cost
        return cost
        

    
    def __execute_supervised_planning(self) -> None:
        self.__exec_plan = True

        #print("starting planners")
        # set planners to perform planning in parallel
        self.__interpolator.plan(self.__planner_data, self.__goal_result)
        self.__hybrid__astar.plan(self.__planner_data, self.__goal_result)
        self.__overtaker.plan(self.__planner_data, self.__goal_result)
        #self.__rrt_star.plan(self.__planner_data, self.__goal_result)
        self.__path_version = 0

        planning_set_exec = []
        planning_set = [self.__interpolator, self.__overtaker, self.__hybrid__astar, self.__rrt_star]
        planning_set_size = len(planning_set)
        planning_set_exec = [True] * (2 * planning_set_size)
        
        best_res = None
        best_path_quality_cost: float = float('inf')
        
        while (not self._check_timeout()):
            if not self.__check_someone_is_planning(planning_set_exec): break
            
            for i in range(planning_set_size):
                if planning_set_exec[i]:
                    if i == 2:
                        pass
                    q = self.__check_planner_coarse(planning_set[i])
                    too_close, exec, res, path_quality = q
                    if exec: continue
                    planning_set_exec[i] = False
                    
                    if path_quality is None:
                        continue
                    cost = self.__curve_quality_cost(path_quality, self.__goal_result.goal)
                    
                    if best_res is None:
                        best_res = res
                        best_path_quality_cost = cost
                        self.__path_version = 1
                        #self.__exec_plan = True
                        print("ready to exec plan")
                    elif cost < best_path_quality_cost:
                        best_res = res
                        best_path_quality_cost = cost
                        self.__path_version += 1
                        #self.__exec_plan = True
                        print(f"a new path is ready")
                
                elif planning_set_exec[planning_set_size + i]:
                    too_close, exec, res, path_quality = self.__check_planner_optim(planning_set[i])
                    if exec: continue
                    planning_set_exec[planning_set_size + i] = False
                    
                    if path_quality is None:
                        continue

                    cost = self.__curve_quality_cost(path_quality, self.__goal_result.goal)
                    
                    if best_res is None:
                        best_res = res
                        best_path_quality_cost = cost
                        self.__path_version = 1
                        #self.__exec_plan = True
                        print("ready to exec plan")
                    elif cost < best_path_quality_cost:
                        best_res = res
                        best_path_quality_cost = cost
                        self.__path_version += 1
                        #self.__exec_plan = True
                        print(f"a new path is ready, from (optim) planner {planning_set[i].__name__}")
                    
            
        self.__plan_result = best_res
        self.__exec_plan = False
        #print("terminating planners")
        self.cancel()


