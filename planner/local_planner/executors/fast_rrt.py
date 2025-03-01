import ctypes.util
import math
import ctypes
import numpy as np
from threading import Thread

from model.physical_parameters import PhysicalParameters
from model.planning_data import PlanningData, PlanningResult
from model.waypoint import Waypoint
from planner.goal_point_discover import GoalPointDiscoverResult
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor

DIST_TO_GOAL_TOLERANCE = 5.0
MAX_STEP_SIZE = 30.0

class FastRRT(LocalPathPlannerExecutor):
    __ptr: ctypes.c_void_p
    __searchThr: Thread
    __expected_velocity_meters_s: float
    
    def __init__(self, 
                 timeout_ms: int,
                 expected_velocity_meters_s: float):
        
        FastRRT.setup_cpp_lib(None)
        
        self.__ptr = FastRRT.lib.fastrrt_initialize(
            PhysicalParameters.OG_WIDTH,
            PhysicalParameters.OG_HEIGHT,
            PhysicalParameters.OG_REAL_WIDTH,
            PhysicalParameters.OG_REAL_HEIGHT,
            math.radians(PhysicalParameters.MAX_STEERING_ANGLE),
            PhysicalParameters.VEHICLE_LENGTH_M,
            timeout_ms,
            MAX_STEP_SIZE,
            DIST_TO_GOAL_TOLERANCE)
        
        self.__searchThr = None
        self.__expected_velocity_meters_s = expected_velocity_meters_s
        
    
    @classmethod
    def setup_cpp_lib(cls, lib_path: str) -> None:
        if hasattr(FastRRT, "lib"):
            return
        
        ctypes.CDLL("/usr/local/lib/driveless/libdriveless.so", mode = ctypes.RTLD_GLOBAL)
        if lib_path is None:
            FastRRT.lib = ctypes.CDLL("/usr/local/lib/driveless/libfastrrt.so", mode = ctypes.RTLD_GLOBAL)
        else:
            FastRRT.lib = ctypes.CDLL(lib_path, mode = ctypes.RTLD_GLOBAL)

        FastRRT.lib.fastrrt_initialize.restype = ctypes.c_void_p
        FastRRT.lib.fastrrt_initialize.argtypes = [
            ctypes.c_int, # width
            ctypes.c_int, # height
            ctypes.c_float, # perceptionWidthSize_m
            ctypes.c_float, # perceptionHeightSize_m
            ctypes.c_float, # maxSteeringAngle_rad
            ctypes.c_float, # vehicleLength
            ctypes.c_int, # timeout_ms
            ctypes.c_float, # maxPathSize
            ctypes.c_float  # distToGoalTolerance
        ]
        
        FastRRT.lib.fastrrt_destroy.restype = None
        FastRRT.lib.fastrrt_destroy.argtypes = [
            ctypes.c_void_p
        ]

        FastRRT.lib.is_planning.restype = ctypes.c_bool
        FastRRT.lib.is_planning.argtypes = [
            ctypes.c_void_p
        ]
        
        FastRRT.lib.set_plan_data.restype = None
        FastRRT.lib.set_plan_data.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,      # goal_x
            ctypes.c_int,      # goal_z
            ctypes.c_float,    # heading_rad
            ctypes.c_float     # velocity_m_s             
        ]

        FastRRT.lib.run.restype = None
        FastRRT.lib.run.argtypes = [
            ctypes.c_void_p,
        ]

        FastRRT.lib.optimize.restype = None
        FastRRT.lib.optimize.argtypes = [
            ctypes.c_void_p,
        ]

        FastRRT.lib.cancel.restype = None
        FastRRT.lib.cancel.argtypes = [
            ctypes.c_void_p,
        ]

        FastRRT.lib.goal_reached.restype = ctypes.c_bool
        FastRRT.lib.goal_reached.argtypes = [
            ctypes.c_void_p,
        ]

        FastRRT.lib.get_planned_path.restype = ctypes.POINTER(ctypes.c_float)
        FastRRT.lib.get_planned_path.argtypes = [
            ctypes.c_void_p,
        ]

        FastRRT.lib.release_planned_path_data.restype = None
        FastRRT.lib.release_planned_path_data.argtypes = [
            ctypes.POINTER(ctypes.c_float),
        ]
        
        
    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        
        if goal_result.too_close():
            return
        
        goal: Waypoint = goal_result.goal()
        
        planner_data.og
        
        FastRRT.lib.set_plan_data(
            self.__ptr, 
            goal.x, 
            goal.z, 
            goal.heading, 
            self.__expected_velocity_meters_s
          )
        pass

    def cancel(self) -> None:
        pass

    def is_planning(self) -> bool:
        pass

    def get_result(self) -> PlanningResult:
        pass
    
    def destroy(self) -> None:
        pass
    