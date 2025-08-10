from enum import Enum
from pydriveless import Waypoint
from .curve_quality import CurveData

class PlannerResultType(Enum):
    NONE =0
    VALID = 1
    INVALID_START = 2
    INVALID_GOAL = 3
    INVALID_PATH = 4
    TOO_CLOSE = 5

class PlanningResult:
    planner_name: str
    timeout: bool
    result_type: PlannerResultType
    path: list[Waypoint]
    planning_exec_time_ms: int
    optimize_exec_time_ms: int
    total_exec_time_ms: int
    path_metrics: CurveData
    curve_cost: float


    def __init__(self, planner_name: str):
        self.planner_name = planner_name
        self.result_type = PlannerResultType.NONE
        self.path = None
        self.planning_exec_time_ms = 0
        self.optimize_exec_time_ms = 0
        self.total_exec_time_ms = 0
        self.timeout = False
        self.curve_cost = 0.0
        self.path_metrics = None

    def __str__(self):
        if self.path is None or len(self.path) == 0: 
            path_info = "-"
        else:
            path_info = f"{len(self.path)} waypoints, {self.path[0]} --> {self.path[-1]}"

        timeout_info = ""
        if self.timeout:
            timeout_info = "[timeout]"

        msg = f"[{self.planner_name}] planning result: {self.result_type.name}" + \
                f"\n\tpath: {path_info}\n\tplan time: {self.planning_exec_time_ms:.2f} ms" +\
                f"\n\toptimize_time: {self.optimize_exec_time_ms:.2f} ms" +\
                f"\n\ttotal exec time: {self.total_exec_time_ms:.2f} ms {timeout_info}" +\
                f"\n\tquality cost: {self.curve_cost:.2f}"
        
        # if self.path_metrics is not None:
        #     msg += f"\n\tmetrics:"
        #     msg += f"\n\t\tgoal reached: {self.path_metrics.goal_reached}"
        #     msg += f"\n\t\tjerk: {self.path_metrics.jerk:.2f}"
        #     msg += f"\n\t\tdiscontinuity: {self.path_metrics.tan_discontinuity}"
        #     msg += f"\n\t\tnum loops: {self.path_metrics.num_loops}"
        #     msg += f"\n\t\toptimized? {self.path_metrics.coarse}"
            
        return msg
