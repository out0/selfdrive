from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
import time

from planner.local_planner.executors.interpolator import InterpolatorPlanner
from planner.local_planner.executors.overtaker import OvertakerPlanner
from planner.local_planner.executors.hybridAStar import HybridAStarPlanner
from planner.local_planner.executors.rrtStar2 import RRTPlanner
from planner.goal_point_discover import GoalPointDiscoverResult
import threading
import numpy as np, math, json


class CurveData:
    name: str
    curve: list[Waypoint]
    num_points: int
    jerk: float
    total_length: float
    tan_discontinuity: bool
    coarse: bool
    proc_time_ms: float
    timeout: bool
    goal_reached: bool
    last_p: Waypoint
    
    def __init__(self, 
                 name: str,
                 coarse: bool,
                 curve: list[Waypoint],
                 num_points: int,
                 jerk: float,
                 total_length: float,
                 tan_discontinuity: bool,
                 proc_time_ms: float,
                 num_loops: int,
                 timeout: bool,
                 goal_reached: bool,
                 last_p: Waypoint):
        self.name = name
        self.curve = curve
        self.num_points = num_points
        self.jerk = jerk
        self.total_length = total_length
        self.tan_discontinuity = tan_discontinuity
        self.coarse = coarse
        self.proc_time_ms = proc_time_ms
        self.timeout = timeout
        self.num_loops = num_loops
        self.goal_reached = goal_reached
        self.last_p = last_p

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "curve" : [],
            "type": "coarse" if self.coarse else "optim",
            "num_points": self.num_points,
            "jerk": self.jerk,
            "total_length": self.total_length,
            "tan_discontinuity": self.tan_discontinuity,
            "proc_time_ms": self.proc_time_ms,
            "timeout": self.timeout,
            "num_loops": self.num_loops,
            "goal_reached": self.goal_reached
            
        }
        l = len(self.curve)
        for i in range(l):
            data["curve"].append(f"({int(self.curve[i].x)}, {int(self.curve[i].z)}, {math.degrees(self.curve[i].heading):0.2f})")
            
        return f"{json.dumps(data)}\n"
    
    def to_csv_header(self) -> str:
        return "\"name\";\"num_points\"; \"jerk\"; \"total_length\"; \"tan_discontinuity\"; \"coarse\"; \"proc_time_ms\"; \"timeout\"; \"num_loops\";\"goal_reached\"\n"
    
    def to_csv(self) -> str:
        disc = "yes" if self.tan_discontinuity else "no"
        curve_type = "coarse" if self.coarse else "optim"
        timeout = "yes" if self.timeout else "no"
        goal_reached = "yes" if self.goal_reached else "no"
        return f"\"{self.name}\";\"{self.num_points}\"; \"{self.jerk}\"; \"{self.total_length}\"; \"{disc}\"; \"{curve_type}\"; \"{self.proc_time_ms}\" \"{timeout}\"; \"{self.num_loops}\";\"{goal_reached}\"\n"


class CurveAssessment:
    def __compute_curve_length(curve: list[Waypoint]) -> float:
        size: float = 0
        for i in range(1, len(curve)):
            size += Waypoint.distance_between(curve[i-1], curve[i])
        return size

    
    def assess_curve(curve: list[Waypoint], start_heading: float, compute_heading: bool = True) -> CurveData:
        return CurveData(
            name=None,
            coarse=False,
            goal_reached=False,
            timeout=False,
            proc_time_ms=0.0,
            num_loops=0,
            curve=curve,
            num_points=len(curve),
            total_length=CurveAssessment.__compute_curve_length(curve),
            jerk=CurveAssessment.__compute_jerk(curve),
            tan_discontinuity=CurveAssessment.__tangential_discontinuity(curve, window_side=4, threshold=15, start_heading=start_heading, compute_heading=compute_heading),
            last_p=curve[-1]
        )

    def __curve_heading(p1: Waypoint, p2:Waypoint) -> tuple[float, bool]:
        if p1.x == p2.x and p1.z == p2.z: return 0.0, False

        dx = p2.x - p1.x
        dz = p2.z - p1.z

        h = (math.pi / 2) - math.atan2(-dz, dx)
        if h > math.pi:
            h = h - 2 * math.pi

        return h, True

    def __curve_mean_heading(curve: np.ndarray, pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        l = len(curve)

        if i < 0:
            j = j - i
            i = 0
        if j >= l:
            i = i - (j - l) - 1
            j = l - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p, valid = CurveAssessment.__curve_heading(curve[k], curve[pos])
            if not valid and debug:
                print (f"not valid for {k}, {pos}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({k}, {pos}) = {math.degrees(p)} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p, valid = CurveAssessment.__curve_heading(curve[pos], curve[k])
            if not valid and debug:
                print (f"not valid for {pos}, {k}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({pos}, {k}) = {math.degrees(p)} current mean: {h/count}")

        return h/count

    def __tangential_discontinuity(curve: np.ndarray, window_side: int=2, threshold: float = 15, start_heading: float = 0.0, compute_heading: bool = True):
        #a_before = curve_mean_heading(curve, 0, window_side=window_side)
        a_before = start_heading
        threshold_rad = math.radians(threshold)
        for i in range(1, len(curve)):
            if compute_heading:
                a = CurveAssessment.__curve_mean_heading(curve, i, window_side=window_side)
            else:
                a = curve[i, 2]
            if abs(a - a_before) > threshold_rad:
                return True
                #print(f"pos: {i} {(curve[i][0], curve[i][1])} spike: {a_before} -> {a}")
            a_before = a
        return False


    def __derivate_p(p1: Waypoint, p2: Waypoint) -> float:
        dx = p2.x - p1.x
        dz = p2.z - p1.z
        return math.atan2(dz,dx)

    def __derivate_curve_p(curve: list[Waypoint], pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        if i < 0:
            j = j - i
            i = 0
        if j >= len(curve):
            i = i - (j - len(curve)) - 1
            j = len(curve) - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p = CurveAssessment.__derivate_p(curve[k], curve[pos])
            h += p
            count += 1
            if debug:
                print (f"h({k}, {pos}) = {p} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p = CurveAssessment.__derivate_p(curve[pos], curve[k])        
            h += p
            count += 1
            if debug:
                print (f"h({pos}, {k}) = {p} current mean: {h/count}")

        return h/count

    def __gradient(curve: list[Waypoint], window_side: int = 2, debug = False):
        dev_curve = []
        for i in range (0, len(curve)):
            dev_curve.append(CurveAssessment.__derivate_curve_p(curve, pos=i, window_side=window_side, debug=debug))
        return dev_curve

    def __sum(vals: list) -> float:
        v = 0
        for i in range (len(vals)):
            v += abs(vals[i])
        return v

    def __compute_jerk(curve):
        v = CurveAssessment.__gradient(curve, window_side=4)
        a = np.gradient(v, 1)
        j = np.gradient(a, 1)
        return abs(CurveAssessment.__sum(j))



    

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

        
        path_telemetry = []

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


