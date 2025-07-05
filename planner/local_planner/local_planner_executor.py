from vision.occupancy_grid_cuda import OccupancyGrid
from model.planning_data import PlanningResult, PlanningData
import time, cv2
from model.waypoint import Waypoint
from planner.goal_point_discover import GoalPointDiscoverResult


class LocalPathPlannerExecutor:
    
    __timeout: int
    __max_exec_time_ms: int
    _exec_start: float
    
    def __init__(self, 
                 max_exec_time_ms: int) -> None:
        
        self.__max_exec_time_ms = max_exec_time_ms
        self.__timeout = -1
        self._exec_start = 0

    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        pass

    def cancel(self) -> None:
        pass

    def is_planning(self) -> bool:
        pass
    
    def is_optimizing(self) -> bool:
        return False

    def get_result(self) -> PlanningResult:
        pass
    
    def get_path_version(self) -> int:
        return 1
    
    def destroy(self) -> None:
        pass
    
    def _rst_timeout(self) -> None:
        self.__timeout = time.time()
    
    def _check_timeout(self) -> bool:
        if (self.__max_exec_time_ms < 0): return False
        
        if (self.__timeout < 0):
            self._rst_timeout()
            return False
        
        return 1000*(time.time() - self.__timeout) >= self.__max_exec_time_ms
    
    def _get_spent_time(self) -> int:
        if (self.__timeout < 0):
            self._rst_timeout()
            return 0
        
        return 1000*(time.time() - self.__timeout)
    
    
    def _get_max_exec_time_ms(self) -> int:
        return self.__max_exec_time_ms
    
    
    def set_exec_started(self) -> None:
        self._exec_start = time.time()
        
    def get_execution_time(self) -> int:
        return 1000 * (time.time() - self._exec_start)
    
    
    def dump_log(self, log_name: str, og: OccupancyGrid, path: list[Waypoint]) -> None:
        f = og.get_color_frame()
        for p in self._result.path:
            if p.z > og.height() or p.x > og.width():
                continue
            f[p.z, p.x, :] = [255, 255, 255]
            cv2.imwrite(f"{log_name}.png", f)
            
        with open(f"{log_name}_path.log", "w") as log:
            for p in path:
                log.write(f"{p}\n")
            