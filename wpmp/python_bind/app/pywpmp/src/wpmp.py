import ctypes.util
import ctypes
import numpy as np
import os
from pydriveless import EgoParams, SearchParams, Waypoint, angle
from pydriveless import LocalPlanner

class WPMP (LocalPlanner):
     __ptr: ctypes.c_void_p
        
     def __init__(self, ego_params: EgoParams, smart: bool):
          super().__init__("WPMP")
          WPMP.setup_cpp_lib()

          path_costs = ego_params.segmentation_class_costs
          perception_width_m, perception_height_m = ego_params.search_frame_physical_dimensions
          max_steering_angle_deg = ego_params.max_steering_angle.deg()
          vehicle_length_m = ego_params.vehicle_length_m
          max_curvature = ego_params.max_curvature
        
          costs = np.ascontiguousarray(np.concatenate(([path_costs.shape[0]], path_costs)), dtype=np.float32)
          w, h = ego_params.search_frame_dimensions
          lb_x, lb_z = ego_params.ego_lower_bound
          ub_x, ub_z = ego_params.ego_upper_bound

          self.__ptr = WPMP.lib.wpmp_initialize(
                 w, h,
                 perception_width_m,
                 perception_height_m,
                 max_steering_angle_deg,
                 vehicle_length_m,
                 lb_x, lb_z,
                 ub_x, ub_z,
                 costs,
                 max_curvature,
                 smart)

     def __del__(self) -> None:
          if hasattr(WPMP, "lib"):
               WPMP.lib.WPMP_destroy(self.__ptr)

     @classmethod
     def setup_cpp_lib(cls) -> None:
          if hasattr(WPMP, "lib"):
               return
        
          lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libWPMP.so")
          lib_path_driveless = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
          WPMP.lib = ctypes.CDLL(lib_path)
          ctypes.CDLL(lib_path_driveless, mode=ctypes.RTLD_GLOBAL)
          
          WPMP.lib.wpmp_initialize.restype = ctypes.c_void_p
          WPMP.lib.wpmp_initialize.argtypes = [
            ctypes.c_int,     # width
            ctypes.c_int,     # height
            ctypes.c_float,   # perceptionWidthSize_m
            ctypes.c_float,   # perceptionHeightSize_m
            ctypes.c_float,   # maxSteeringAngle_rad
            ctypes.c_float,   # vehicleLength
            ctypes.c_int,     # lowerBound_x
            ctypes.c_int,     # lowerBound_z
            ctypes.c_int,     # upperBound_x
            ctypes.c_int,     # upperBound_z
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), # segmentationClassCost
            ctypes.c_float,   # max_curvature
            ctypes.c_bool     # smart
          ]
        
          WPMP.lib.wpmp_destroy.restype = None
          WPMP.lib.wpmp_destroy.argtypes = [
             ctypes.c_void_p
          ]

          WPMP.lib.set_plan_data.restype = None
          WPMP.lib.set_plan_data.argtypes = [
               ctypes.c_void_p,
               ctypes.c_void_p,   # cuda_ptr
               ctypes.c_int,      # start_x
               ctypes.c_int,      # start_z
               ctypes.c_float,    # start_heading_rad
               ctypes.c_int,      # goal_x
               ctypes.c_int,      # goal_z
               ctypes.c_float,    # goal_heading_rad
               ctypes.c_float,    # velocity_m_s             
               ctypes.c_int,      # min_dist_x
               ctypes.c_int,       # min_dist_z
               ctypes.c_int,     # timeout_ms
               ctypes.c_float,   # maxPathSize
               ctypes.c_float,   # distToGoalTolerance
               ctypes.c_float,   # headingErrorTolerance_rad

          ]

          WPMP.lib.goal_reached.restype = ctypes.c_bool
          WPMP.lib.goal_reached.argtypes = [
               ctypes.c_void_p,
          ]

          WPMP.lib.initialize.restype = None
          WPMP.lib.initialize.argtypes = [
               ctypes.c_void_p,
               ctypes.c_bool       # copyIntrinsicCostsFromFrame
          ]

          WPMP.lib.planning_loop.restype = ctypes.c_bool
          WPMP.lib.planning_loop.argtypes = [
               ctypes.c_void_p
          ]          

          WPMP.lib.path_optimize_loop.restype = ctypes.c_bool
          WPMP.lib.path_optimize_loop.argtypes = [
               ctypes.c_void_p,
          ]          
          
          WPMP.lib.get_planned_path.restype = ctypes.POINTER(ctypes.c_float)
          WPMP.lib.get_planned_path.argtypes = [
               ctypes.c_void_p,
          ]
          
          WPMP.lib.get_interpolated_planned_path.restype = ctypes.POINTER(ctypes.c_float)
          WPMP.lib.get_interpolated_planned_path.argtypes = [
               ctypes.c_void_p,
          ]
                   
          WPMP.lib.ideal_curve.restype = ctypes.POINTER(ctypes.c_float)
          WPMP.lib.ideal_curve.argtypes = [
               ctypes.c_void_p,
               ctypes.c_int,       # goal_x
               ctypes.c_int,       # goal_z
               ctypes.c_float,     # goal_heading
          ]

          WPMP.lib.release_planned_path_data.restype = None
          WPMP.lib.release_planned_path_data.argtypes = [
               ctypes.POINTER(ctypes.c_float),
          ]
          
          WPMP.lib.export_graph_nodes.restype = ctypes.POINTER(ctypes.c_int)
          WPMP.lib.export_graph_nodes.argtypes = [
               ctypes.c_void_p,
          ]
          
          WPMP.lib.release_export_graph_nodes.restype = None
          WPMP.lib.release_export_graph_nodes.argtypes = [
               ctypes.POINTER(ctypes.c_int),
          ]
          
          WPMP.lib.compute_region_debug_performance.restype = None
          WPMP.lib.compute_region_debug_performance.argtypes = [
               ctypes.c_void_p
          ]

          WPMP.lib.save_current_graph_state.restype = None
          WPMP.lib.save_current_graph_state.argtypes = [
               ctypes.c_void_p,
               ctypes.c_char_p     # filename
          ]
          
          WPMP.lib.load_graph_state.restype = None
          WPMP.lib.load_graph_state.argtypes = [
               ctypes.c_void_p,
               ctypes.c_char_p     # filename
          ]          
        
         

     def set_plan_data(self, search_params: SearchParams) -> bool:          
          frame = search_params.frame.get_cuda_ptr()
          start = search_params.start.x, search_params.start.z, search_params.start.heading.rad()
          goal = search_params.goal.x, search_params.goal.z, search_params.goal.heading.rad()
          velocity_m_s = search_params.velocity_m_s
          min_dist = search_params.min_distance
          timeout_ms = search_params.timeout_ms
          max_path_size_px = search_params.max_path_size_px
          dist_to_goal_tolerance_px = search_params.distance_to_goal_tolerance_px

          return WPMP.lib.set_plan_data(
            self.__ptr, 
            frame,
            start[0],
            start[1],
            start[2],
            goal[0],
            goal[1],
            goal[2],
            velocity_m_s,
            min_dist[0],
            min_dist[1],
            timeout_ms,
            max_path_size_px,
            dist_to_goal_tolerance_px,
            search_params.heading_error_tolerance.rad()
          )
   
     def initialize(self, copy_intrinsic_costs_from_frame: bool = False) -> None:
          WPMP.lib.initialize(self.__ptr, copy_intrinsic_costs_from_frame)
     
     def planning_loop(self) -> bool:
          return WPMP.lib.planning_loop(self.__ptr)
        
     def path_optimize_loop(self) -> bool:
          return WPMP.lib.path_optimize_loop(self.__ptr)
     
     def goal_reached(self) -> bool:
          return WPMP.lib.goal_reached(self.__ptr)     
     
     # def __convert_planned_path(self, ptr: ctypes.c_void_p) -> np.ndarray:
     #      size = int(ptr[0])
     #      if size == 0:
     #           return None
          
     #      res = np.zeros((size, 3), dtype=np.float32)
     #      for i in range(size):
     #           pos = 3*i + 1
     #           res[i, 0] = float(ptr[pos])
     #           res[i, 1] = float(ptr[pos + 1])
     #           res[i, 2] = float(ptr[pos + 2])
     #      return res

     def __convert_planned_path(self, ptr: ctypes.c_void_p) -> tuple[list[Waypoint], float]:
          size = int(ptr[0])
          cost = float(ptr[1])
          if size == 0:
               return None, -1
          
          res = []
          for i in range(size):
               pos = 3*i + 2
               res.append(Waypoint(
                    x=int(ptr[pos]),
                    z=int(ptr[pos + 1]),
                    heading=angle.new_rad(ptr[pos + 2])))
          return res, cost
     
     def get_planned_path(self) -> tuple[list[Waypoint], float]:
          ptr = WPMP.lib.get_planned_path(self.__ptr)         
          res, cost = self.__convert_planned_path(ptr)
          WPMP.lib.release_planned_path_data(ptr)
          return res, cost

     def get_interpolated_planned_path(self) -> tuple[list[Waypoint], float]:
          ptr = WPMP.lib.get_interpolated_planned_path(self.__ptr)
          res, cost = self.__convert_planned_path(ptr)
          WPMP.lib.release_planned_path_data(ptr)
          return res, cost


     # def interpolate_planned_path_p(self, path: np.ndarray) -> np.ndarray:          
     #      size = path.shape[0]
     #      path = path.reshape(3*size)
     #      ptr = WPMP.lib.interpolate_planned_path_p(self.__ptr, path, 3*size) 
     #      path.reshape((size, 3))
     #      res = self.__convert_planned_path(ptr)
     #      WPMP.lib.release_planned_path_data(ptr)
     #      return res
     
     def build_ideal_curve(self, goal_x: int, goal_z:int, goal_heading: float) -> np.ndarray:
          ptr = WPMP.lib.ideal_curve(self.__ptr, goal_x, goal_z, goal_heading)
          res, _ = self.__convert_planned_path(ptr)
          WPMP.lib.release_planned_path_data(ptr)
          return res
     