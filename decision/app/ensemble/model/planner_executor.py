
import time
from .planning_data import PlanningData
from .planning_result import PlanningResult, PlannerResultType
from threading import Thread
from pydriveless import Waypoint

class LocalPlannerExecutor:
    __name: str
    __max_exec_time_ms: int
    __exec_start: float
    __is_planning: bool
    __is_optimizing: bool
    __is_running: bool
    __can_run: bool
    __loop_thr: Thread
    __planning_data: PlanningData
    _planning_result: PlanningResult
        
    def __init__(self, name: str, max_exec_time_ms: int):
        self.__name = name
        self.__max_exec_time_ms = max_exec_time_ms
        self.__exec_start = -1
        self.__can_run = True
        self.__is_planning = False
        self.__is_optimizing = False
        self.__loop_thr = None
        self.__is_running = False
        self._planning_result = PlanningResult(self.__name)        
    
    def cancel(self) -> None:
        if self.__can_run:
            #print("terminating process")
            self.__can_run = False
            if self.__loop_thr is not None:
                self.__loop_thr.join()
                self.__loop_thr = None
        else:
            #print("process was terminated already")
            self.__can_run = False
            self.__loop_thr = None

        self._planning_result = PlanningResult(self.__name) 

    def get_planner_name(self) -> str:
        return self.__name

    def is_planning(self) -> bool:
        return self.__is_planning
    
    def is_optimizing(self) -> bool:
        return self.__is_optimizing
    
    def new_path_available(self) -> bool:
        return self._planning_result.result_type == PlannerResultType.VALID

    def timeout (self) -> bool:
        return self._planning_result.timeout
       
    def __rst_timeout(self) -> None:
        self.__exec_start = time.time()
    
    def _check_timeout(self) -> bool:
        if (self.__max_exec_time_ms < 0): return False
        if (self.__exec_start <= 0):
            self.__rst_timeout()
            return False
        return 1000*(time.time() - self.__exec_start) >= self.__max_exec_time_ms
    
    def get_execution_time(self) -> int:
        return 1000 * (time.time() - self.__exec_start)
    
    def plan (self, data: PlanningData, run_in_main_thread: bool = False) -> None:
        self.cancel()
        self.__can_run = True
        self.__is_planning = True
        self.__is_optimizing = True
        self.__is_running = True
        self.__planning_data = data
        self.__rst_timeout()

        if run_in_main_thread:
            if not self._planning_init(data):
                return
            self._planning_exec()
            return
        
        self.__loop_thr = Thread(target=self._planning_exec)
        self.__loop_thr.start()
    
    def _planning_exec(self) -> None:
        timeout: bool = False
        self.__is_running = True
        
        if self._planning_init(self.__planning_data):
            while not timeout and self.__can_run and self.__is_planning:
                timeout = self._check_timeout()
                if timeout: continue
                self.__is_planning = self._loop_plan(self.__planning_data)

            self._planning_result.planning_exec_time_ms = self.get_execution_time()
            
            while not timeout and self.__can_run and self.__is_optimizing:
                timeout = self._check_timeout()
                if timeout: continue
                self.__is_optimizing = self._loop_optimize(self.__planning_data)

            self.__is_running = False
        
        self.__can_run = False
        self.__is_planning = False
        self.__is_optimizing = False
        self._planning_result.timeout = timeout
        self._planning_result.total_exec_time_ms = self.get_execution_time()
        self._planning_result.optimize_exec_time_ms = self._planning_result.total_exec_time_ms - self._planning_result.planning_exec_time_ms

    def _set_planning_result(self, result_type: PlannerResultType, path: list[Waypoint]):
        self._planning_result.path = path
        self._planning_result.result_type = result_type

    def is_running(self) -> bool:
        self.__is_running
    
    def get_result(self) -> PlanningResult:
        return self._planning_result
    
    def get_max_exec_time_ms(self) -> int:
        return self.__max_exec_time_ms
    
    #----------------------------------------------------------
    #
    #               Extend and Implement These
    #
    #----------------------------------------------------------
    def _planning_init(self, planning_data: PlanningData) -> bool:
        return True

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        pass
    
    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        pass
    
