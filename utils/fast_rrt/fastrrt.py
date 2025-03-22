import ctypes.util
import ctypes
import numpy as np

class FastRRT:
     __ptr: ctypes.c_void_p
        
     def __init__(self, 
                 width: int, 
                 height: int,
                 perception_width_m: float,
                 perception_height_m: float,
                 max_steering_angle_deg : float,
                 vehicle_length_m: float,
                 timeout_ms: int,
                 min_dist_x: int,
                 min_dist_z: int,
                 lower_bound_x: int,
                 lower_bound_z: int,
                 upper_bound_x: int,
                 upper_bound_z: int,
                 max_path_size_px: float = 30.0,
                 dist_to_goal_tolerance_px: float = 5.0,
                 libdir = None
                 ):
          FastRRT.setup_cpp_lib(libdir)
        
          self.__ptr = FastRRT.lib.fastrrt_initialize(
                 width, 
                 height,
                 perception_width_m,
                 perception_height_m,
                 max_steering_angle_deg,
                 vehicle_length_m,
                 timeout_ms,
                 min_dist_x,
                 min_dist_z,
                 lower_bound_x,
                 lower_bound_z,
                 upper_bound_x,
                 upper_bound_z,
                 max_path_size_px,
                 dist_to_goal_tolerance_px)
    
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
               ctypes.c_int,      # goal_x
               ctypes.c_int,      # goal_z
               ctypes.c_float,    # heading_rad
               ctypes.c_float     # velocity_m_s             
          ]

          FastRRT.lib.goal_reached.restype = ctypes.c_bool
          FastRRT.lib.goal_reached.argtypes = [
               ctypes.c_void_p,
          ]

          FastRRT.lib.search_init.restype = None
          FastRRT.lib.search_init.argtypes = [
               ctypes.c_void_p,
          ]

          FastRRT.lib.loop.restype = ctypes.c_bool
          FastRRT.lib.loop.argtypes = [
               ctypes.c_void_p,
          ]          

          FastRRT.lib.loop_optimize.restype = ctypes.c_bool
          FastRRT.lib.loop_optimize.argtypes = [
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
          
          FastRRT.lib.export_graph_nodes.restype = ctypes.POINTER(ctypes.c_int)
          FastRRT.lib.export_graph_nodes.argtypes = [
               ctypes.c_void_p,
          ]
          
          FastRRT.lib.release_export_graph_nodes.restype = None
          FastRRT.lib.release_export_graph_nodes.argtypes = [
               ctypes.POINTER(ctypes.c_int),
          ]

     def set_plan_data(self, cuda_ptr: ctypes.c_void_p, goal_x: int, goal_z: int, heading_rad: float, velocity_m_s: float) -> bool:
          return FastRRT.lib.set_plan_data(
            self.__ptr, 
            cuda_ptr,
            goal_x,
            goal_z,
            heading_rad,
            velocity_m_s
          )
   
     def search_init(self) -> None:
          FastRRT.lib.search_init(self.__ptr)
     
     def loop(self) -> bool:
          return FastRRT.lib.loop(self.__ptr)
        
     def loop_optimize(self) -> bool:
          return False
          return FastRRT.lib.loop_optimize(self.__ptr)
     
     def goal_reached(self) -> bool:
          return FastRRT.lib.goal_reached(self.__ptr)     
         
     def get_planned_path(self) -> np.ndarray:
          ptr = FastRRT.lib.get_planned_path(self.__ptr)

          #data = ptr.contents
          size = int(ptr[0])
          if size == 0:
               FastRRT.lib.release_planned_path_data(ptr)
               return None
          
          res = np.zeros((size, 3), dtype=np.float32)
          for i in range(size):
               pos = 3*i + 1
               res[i, 0] = float(ptr[pos])
               res[i, 1] = float(ptr[pos + 1])
               res[i, 2] = float(ptr[pos + 2])
               # res[i, 0] = float(data[pos])
               # res[i, 1] = float(data[pos + 1])
               # res[i, 2] = float(data[pos + 2])
          
          FastRRT.lib.release_planned_path_data(ptr)
          return res

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
          
     