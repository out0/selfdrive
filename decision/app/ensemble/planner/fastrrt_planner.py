
from pydriveless import CoordinateConverter
from pyfastrrt import FastRRT
from ..model.planner_executor import LocalPlannerExecutor
from ..model.planning_result import PlanningResult, PlannerResultType
from ..model.planning_data import PlanningData
from ..model.physical_paramaters import PhysicalParameters
import math
import numpy as np
import random


class FastRRT(LocalPlannerExecutor):
    _map_coordinate_converter: CoordinateConverter
    _fastrrt: FastRRT
    _max_exec_time_ms: int
    _max_path_size_px: int
    _dist_to_goal_tolerance_px: int
 
    
    def __init__(self, map_coordinate_converter: CoordinateConverter,
                 max_exec_time_ms: int, 
                 max_path_size_px: int = 20, 
                 dist_to_goal_tolerance_px: int = 20
                 ):
        
        super().__init__("FastRRT", max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self._max_exec_time_ms = max_exec_time_ms
        self._max_path_size_px = max_path_size_px
        self._dist_to_goal_tolerance_px = dist_to_goal_tolerance_px
   
    def _planning_init(self, planning_data: PlanningData) -> bool:
        self._fastrrt = FastRRT(
            search_frame = planning_data.og(),
            perception_width_m = PhysicalParameters.OG_REAL_WIDTH,
            perception_height_m = PhysicalParameters.OG_REAL_HEIGHT,
            max_steering_angle_deg = PhysicalParameters.MAX_STEERING_ANGLE,
            vehicle_length_m = PhysicalParameters.VEHICLE_LENGTH_M,
            timeout_ms = self._max_exec_time_ms,
            min_dist_x = PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            min_dist_z = PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            path_costs = PhysicalParameters.SEGMENTATION_CLASS_COST,
            max_path_size_px = self._max_path_size_px,
            dist_to_goal_tolerance_px = self._dist_to_goal_tolerance_px
        )

        self._fastrrt.set_plan_data(
            cuda_ptr=planning_data.og().get_cuda_ptr(),
            start=(planning_data.start().x, planning_data.start().z, planning_data.start().heading.rad()),
            goal=(planning_data.local_goal().x, planning_data.local_goal().z, planning_data.local_goal().heading.rad()), 
            velocity_m_s=planning_data.velocity)
        
        self._fastrrt.search_init()
        return True

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        return self._fastrrt.loop(False)

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        self._fastrrt.optimize_path()
        return False

    
