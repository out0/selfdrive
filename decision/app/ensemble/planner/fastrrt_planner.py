
from pydriveless import EgoParams, SearchParams
from pyfastrrt import FastRRT
from ..model.planner_executor import LocalPlannerExecutor
from ..model.planning_result import PlanningResult, PlannerResultType
from ..model.planning_data import PlanningData
from ..model.physical_paramaters import PhysicalParameters
import math
import numpy as np
import random


class FastRRTPlanner(LocalPlannerExecutor):
    _fastrrt: FastRRT
    _pre_process_data: bool
    _smart_expansion: bool
    
    def __init__(self, ego_params: EgoParams, pre_process_data: bool = True, smart_expansion: bool = True):
        
        super().__init__("FastRRT", ego_params)
        self._fastrrt = FastRRT(ego_params)
        self._pre_process_data = pre_process_data
        self._smart_expansion = smart_expansion
   
    def _planning_init(self, search_params: SearchParams) -> bool:
        
        self._fastrrt.set_plan_data(search_params)
        self._fastrrt.search_init()
        
        if self._pre_process_data:
            frame = search_params.frame
            frame.process_safe_distance_zone(min_distance=search_params.min_distance, compute_vectorized=False)
            frame.process_distance_to_goal(search_params.goal.x, search_params.goal.z)
            
        return True

    def _loop_plan(self) -> bool:
        return self._fastrrt.loop(self._smart_expansion)

    def _loop_optimize(self) -> bool:
        return self._fastrrt.path_optimize()

    
