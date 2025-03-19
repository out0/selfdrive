import ctypes.util
import ctypes
import numpy as np
from utils.cudac.cuda_frame import CudaFrame

class CudaGraph:
    __ptr: ctypes.c_void_p
    __width: int
    __height: int
            
    def __init__(self, 
                 width: int, 
                 height: int,
                 perception_width_m: float,
                 perception_height_m: float,
                 max_steering_angle_deg : float,
                 vehicle_length_m: float,
                 min_dist_x: int,
                 min_dist_z: int,
                 lower_bound_x: int,
                 lower_bound_z: int,
                 upper_bound_x: int,
                 upper_bound_z: int,
                 libdir = None
                 ):
          CudaGraph.setup_cpp_lib(libdir)
        
          self.__ptr = CudaGraph.lib.cudagraph_initialize(
                 width, 
                 height,
                 perception_width_m,
                 perception_height_m,
                 max_steering_angle_deg,
                 vehicle_length_m,
                 min_dist_x,
                 min_dist_z,
                 lower_bound_x,
                 lower_bound_z,
                 upper_bound_x,
                 upper_bound_z)
          
          self.__width = width
          self.__height = height
    
    def __del__(self) -> None:
          if hasattr(CudaGraph, "lib"):
               CudaGraph.lib.cudagraph_destroy(self.__ptr)

    @classmethod
    def setup_cpp_lib(cls, lib_path: str) -> None:
          if hasattr(CudaGraph, "lib"):
               return
        
          ctypes.CDLL("/usr/local/lib/libdriveless-cudac.so", mode = ctypes.RTLD_GLOBAL)
          ctypes.CDLL("/usr/local/lib/driveless/libcuda_utils.so", mode = ctypes.RTLD_GLOBAL)
          if lib_path is None:
               CudaGraph.lib = ctypes.CDLL("/usr/local/lib/driveless/libfastrrt.so", mode = ctypes.RTLD_GLOBAL)
          else:
               CudaGraph.lib = ctypes.CDLL(lib_path, mode = ctypes.RTLD_GLOBAL)

          CudaGraph.lib.cudagraph_initialize.restype = ctypes.c_void_p
          CudaGraph.lib.cudagraph_initialize.argtypes = [
            ctypes.c_int, # width
            ctypes.c_int, # height
            ctypes.c_float, # perceptionWidthSize_m
            ctypes.c_float, # perceptionHeightSize_m
            ctypes.c_float, # maxSteeringAngle_rad
            ctypes.c_float, # vehicleLength
            ctypes.c_int, #minDistance_x
            ctypes.c_int, #minDistance_z
            ctypes.c_int, #lowerBound_x
            ctypes.c_int, #lowerBound_z
            ctypes.c_int, #upperBound_x
            ctypes.c_int #upperBound_z
          ]
        
          CudaGraph.lib.cudagraph_destroy.restype = None
          CudaGraph.lib.cudagraph_destroy.argtypes = [
             ctypes.c_void_p
          ]
          
          CudaGraph.lib.compute_apf.restype = None
          CudaGraph.lib.compute_apf.argtypes = [
              ctypes.c_void_p,  # graph ptr
              ctypes.c_void_p,  # cuda search frame ptr
              ctypes.c_int      # radius
          ]
          
          CudaGraph.lib.get_intrinsic_costs.restype = ctypes.POINTER(ctypes.c_float)
          CudaGraph.lib.get_intrinsic_costs.argtypes = [
              ctypes.c_void_p,  # graph ptr
          ]
          
          CudaGraph.lib.destroy_intrinsic_costs_ptr.restype = None
          CudaGraph.lib.destroy_intrinsic_costs_ptr.argtypes = [
              ctypes.c_void_p,  # costs ptr
          ]

    def compute_apf(self, cuda_ptr: CudaFrame, radius: int):
        CudaGraph.lib.compute_apf(self.__ptr, cuda_ptr.get_cuda_frame(), radius)
    

    def get_intrinsic_costs (self) -> np.ndarray:
        
        costs_ptr = CudaGraph.lib.get_intrinsic_costs(self.__ptr)
        
        res = np.zeros((self.__height, self.__width), dtype=np.float32)
        
        for h in range(self.__height):
            for w in range(self.__width):
                res[h, w] = float(costs_ptr[h * self.__width + w])
        
        CudaGraph.lib.destroy_intrinsic_costs_ptr(costs_ptr)
        return res

     