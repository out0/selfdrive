import ctypes.util
import math
import ctypes
import numpy as np
from threading import Thread

from model.physical_parameters import PhysicalParameters
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from model.waypoint import Waypoint
from planner.goal_point_discover import GoalPointDiscoverResult
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
import cv2

DIST_TO_GOAL_TOLERANCE = 15.0
MAX_STEP_SIZE = 30.0

class FastRRT(LocalPathPlannerExecutor):
     __ptr: ctypes.c_void_p
     __searchThr: Thread
     __plan: bool
     __result: PlanningResult
     __goal_result: GoalPointDiscoverResult
     __planner_data: PlanningData
     __debug_og: np.ndarray
    
     def __init__(self, 
                 timeout_ms: int):
        
        self.__result = None
        FastRRT.setup_cpp_lib(None)
        
        self.__ptr = FastRRT.lib.fastrrt_initialize(
                 PhysicalParameters.OG_WIDTH, 
                 PhysicalParameters.OG_HEIGHT,
                 PhysicalParameters.OG_REAL_WIDTH,
                 PhysicalParameters.OG_REAL_HEIGHT,
                 PhysicalParameters.MAX_STEERING_ANGLE,
                 PhysicalParameters.VEHICLE_LENGTH_M,
                 timeout_ms,
                 PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                 PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                 PhysicalParameters.EGO_LOWER_BOUND.x,
                 PhysicalParameters.EGO_LOWER_BOUND.z,
                 PhysicalParameters.EGO_UPPER_BOUND.x,
                 PhysicalParameters.EGO_UPPER_BOUND.z,
                 30.0,
                 10.0)
        
        self.__searchThr = None
        self.__debug_og = None
        self.__plan = False
    
     def __del__(self) -> None:
        if hasattr(FastRRT, "lib"):
            FastRRT.lib.fastrrt_destroy(self.__ptr)
    
     @classmethod
     def setup_cpp_lib(cls, lib_path: str) -> None:
          if hasattr(FastRRT, "lib"):
               return
        
          ctypes.CDLL("/usr/local/lib/libdriveless-cudac.so", mode = ctypes.RTLD_GLOBAL)
          ctypes.CDLL("/usr/local/lib/driveless/libcuda_utils.so", mode = ctypes.RTLD_GLOBAL)
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
            ctypes.c_int, #minDistance_x
            ctypes.c_int, #minDistance_z
            ctypes.c_int, #lowerBound_x
            ctypes.c_int, #lowerBound_z
            ctypes.c_int, #upperBound_x
            ctypes.c_int, #upperBound_z
            ctypes.c_float, # maxPathSize
            ctypes.c_float  # distToGoalTolerance
          ]
        
          FastRRT.lib.fastrrt_destroy.restype = None
          FastRRT.lib.fastrrt_destroy.argtypes = [
             ctypes.c_void_p
          ]

          FastRRT.lib.set_plan_data.restype = None
          FastRRT.lib.set_plan_data.argtypes = [
               ctypes.c_void_p,
               ctypes.c_void_p,   # cuda_ptr
               ctypes.c_int,      # start_x
               ctypes.c_int,      # start_z
               ctypes.c_float,    # start_heading_rad
               ctypes.c_int,      # goal_x
               ctypes.c_int,      # goal_z
               ctypes.c_float,    # goal_heading_rad
               ctypes.c_float     # velocity_m_s             
          ]

          FastRRT.lib.goal_reached.restype = ctypes.c_bool
          FastRRT.lib.goal_reached.argtypes = [
               ctypes.c_void_p,
          ]

          FastRRT.lib.search_init.restype = None
          FastRRT.lib.search_init.argtypes = [
               ctypes.c_void_p,
               ctypes.c_bool       # copyIntrinsicCostsFromFrame
          ]

          FastRRT.lib.loop.restype = ctypes.c_bool
          FastRRT.lib.loop.argtypes = [
               ctypes.c_void_p,
               ctypes.c_bool       # smartExpansion
          ]          

          FastRRT.lib.loop_optimize.restype = ctypes.c_bool
          FastRRT.lib.loop_optimize.argtypes = [
               ctypes.c_void_p,
          ]          
          
          FastRRT.lib.get_planned_path.restype = ctypes.POINTER(ctypes.c_float)
          FastRRT.lib.get_planned_path.argtypes = [
               ctypes.c_void_p,
          ]
          
          FastRRT.lib.interpolate_planned_path.restype = ctypes.POINTER(ctypes.c_float)
          FastRRT.lib.interpolate_planned_path.argtypes = [
               ctypes.c_void_p,
          ]
          
          FastRRT.lib.interpolate_planned_path_p.restype = ctypes.POINTER(ctypes.c_float)
          FastRRT.lib.interpolate_planned_path_p.argtypes = [
               ctypes.c_void_p,
               np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
               ctypes.c_int32
          ]
          
          FastRRT.lib.ideal_curve.restype = ctypes.POINTER(ctypes.c_float)
          FastRRT.lib.ideal_curve.argtypes = [
               ctypes.c_void_p,
               ctypes.c_int,       # goal_x
               ctypes.c_int,       # goal_z
               ctypes.c_float,     # goal_heading
          ]

          FastRRT.lib.release_planned_path_data.restype = None
          FastRRT.lib.release_planned_path_data.argtypes = [
               ctypes.POINTER(ctypes.c_float),
          ]
          
          FastRRT.lib.export_graph_nodes.restype = ctypes.POINTER(ctypes.c_int)
          FastRRT.lib.export_graph_nodes.argtypes = [
               ctypes.c_void_p,
          ]
          
          FastRRT.lib.release_export_graph_nodes.restype = None
          FastRRT.lib.release_export_graph_nodes.argtypes = [
               ctypes.POINTER(ctypes.c_int),
          ]
          
          FastRRT.lib.compute_region_debug_performance.restype = None
          FastRRT.lib.compute_region_debug_performance.argtypes = [
               ctypes.c_void_p
          ]
        
        
     def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        
        if goal_result.too_close:
            return
        
        self.__goal_result = goal_result
        self.__planner_data = planner_data
        start: Waypoint = goal_result.start
        goal: Waypoint = goal_result.goal
        
        self.__debug_og = planner_data.og.get_color_frame()
        
        FastRRT.lib.set_plan_data(
            self.__ptr, 
            planner_data.og.get_cuda_frame().get_cuda_frame(),
            start.x,
            start.z,
            start.heading,
            goal.x,
            goal.z,
            goal.heading,
            planner_data.velocity
          )
        
        self.__plan = True
        self.set_exec_started()
        FastRRT.lib.search_init(self.__ptr, False)
        planner_data.og.get_cuda_frame().update_frame()
     #    f = planner_data.og.get_cuda_frame().get_frame()
     #    n = np.full(f.shape, fill_value=0.0)
     #    for z in range(f.shape[0]):
     #         for x in range(f.shape[1]):
     #           if f[z, x, 2] == 0.0:
     #                n[z, x] = [0, 0, 0]
     #           else:
     #                n[z, x] = [255, 255, 255]
     #    cv2.imwrite("fast_rrt_debug.png", n)
        
        __searchThr = Thread(target=self.__local_planning)
        __searchThr.start()
            

     def __local_planning(self):
        
          while self.__plan and FastRRT.lib.loop(self.__ptr, False):
               p = self.export_graph_nodes()
               for node in p:
                    self.__debug_og[node[1], node[0]] = [255, 255, 255]
               cv2.imwrite("fast_rrt_debug.png", self.__debug_og)               
               pass

          if not FastRRT.lib.goal_reached(self.__ptr):
               self.__result = PlanningResult(
                    planner_name='FastRRT',
                    ego_location=self.__planner_data.ego_location,
                    goal=self.__planner_data.goal,
                    next_goal=self.__planner_data.next_goal,
                    local_start=self.__goal_result.start,
                    local_goal=self.__goal_result.goal,
                    direction=self.__goal_result.direction,
                    path=None,
                    result_type=PlannerResultType.INVALID_PATH,
                    total_exec_time_ms=self.get_execution_time(),
                    timeout=False
               )
               self.__plan = False
               return
        
          optim_loop_count = 20
          
          p = self.export_graph_nodes()
          loop_count = 0
          while  self.__plan and loop_count < optim_loop_count and FastRRT.lib.loop_optimize(self.__ptr):
               loop_count += 1
        
          self.__result = PlanningResult(
                    planner_name='FastRRT',
                    ego_location=self.__planner_data.ego_location,
                    goal=self.__planner_data.goal,
                    next_goal=self.__planner_data.next_goal,
                    local_start=self.__goal_result.start,
                    local_goal=self.__goal_result.goal,
                    direction=self.__goal_result.direction,
                    path=self.get_planned_path(True),
                    result_type=PlannerResultType.VALID,
                    total_exec_time_ms=self.get_execution_time(),
                    timeout=False                  
                )
        
          self.__plan = False

     def cancel(self) -> None:
        self.__plan = False
        if self.__searchThr != None:
            if self.__searchThr.is_alive():
                self.__searchThr.join()
        
        
        
     def export_graph_nodes(self) -> np.ndarray:
          ptr = FastRRT.lib.export_graph_nodes(self.__ptr)
          
          size = ptr[0]
          if size == 0:
               FastRRT.lib.release_export_graph_nodes(ptr)
               return None
          
          nodes = np.zeros((size, 3), dtype=np.int32)
          
          for i in range(size):
               pos = 3*i + 1
               nodes[i, 0] = ptr[pos]
               nodes[i, 1] = ptr[pos + 1]
               nodes[i, 2] = ptr[pos + 2]
               
          FastRRT.lib.release_export_graph_nodes(ptr)
          return nodes
     

     def __convert_planned_path(self, ptr: ctypes.c_void_p) -> np.ndarray:
          size = int(ptr[0])
          if size == 0:
               return None
          
          res = []
          for i in range(size):
               pos = 3*i + 1
               res.append(Waypoint(
                    x=int(float(ptr[pos])),
                    z=int(float(ptr[pos + 1])),
                    heading=float(ptr[pos + 2])
               ))
          return res
     
     def get_planned_path(self, interpolate: bool = False) -> list[Waypoint]:
          if interpolate:
               ptr = FastRRT.lib.interpolate_planned_path(self.__ptr)
          else:
               ptr = FastRRT.lib.get_planned_path(self.__ptr)
          
          res = self.__convert_planned_path(ptr)
          FastRRT.lib.release_planned_path_data(ptr)
          return res

     def is_planning(self) -> bool:
          return self.__plan

     def get_result(self) -> PlanningResult:
          return self.__result
    
     def destroy(self) -> None:
          self.cancel()
          pass
    