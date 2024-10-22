import ctypes
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from model.waypoint import Waypoint
import math
from enum import Enum

LIBNAME = "/usr/local/lib/libdriveless-cuda-rrt-accel.so"
lib = ctypes.CDLL(LIBNAME)

lib.load_frame.restype = ctypes.c_void_p
lib.load_frame.argtypes = [
    ctypes.c_int, # width
    ctypes.c_int, # height
]

lib.destroy_frame.restype = None
lib.destroy_frame.argtypes = [ctypes.c_void_p]


lib.clear.restype = None
lib.clear.argtypes = [
    ctypes.c_void_p
]

lib.add_point.restype = None
lib.add_point.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int, # Z
    ctypes.c_int, # parent_X
    ctypes.c_int, # parent_Z
    ctypes.c_float, # cost
]

lib.find_best_neighbor.restype = ctypes.POINTER(ctypes.c_int * 3)
lib.find_best_neighbor.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int, # Z
    ctypes.c_float, # radius
]

lib.free_waypoint.restype = None
lib.free_waypoint.argtypes = [
    ctypes.POINTER(ctypes.c_int * 3)
]

lib.count.restype = ctypes.c_uint32
lib.count.argtypes = [
    ctypes.c_void_p, 
]
    

lib.check_in_graph.restype = ctypes.c_bool
lib.check_in_graph.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int # Z    
]
        

class CudaGraph:
    _cuda_graph: ctypes.c_void_p
   
    def __init__ (self, width: int, height: int) -> None:
        self._cuda_frame = None
        self._cuda_frame = lib.load_frame(
            width,
            height)
        
    def __del__(self):
        if self._cuda_frame is None:
            return
        lib.destroy_frame(self._cuda_frame)
        
    def clear(self):
        if self._cuda_frame is None:
            return
        lib.clear(self._cuda_frame)

    def add_point(self, x: int, z: int, parent_x: int, parent_z: int, cost: float):
        if self._cuda_frame is None:
            return
        
        lib.add_point(self._cuda_frame, x, z, parent_x, parent_z, cost)


    def find_best_neighbor(self, x: int, z: int, radius: float) -> tuple[int, int]:
        if self._cuda_frame is None:
            return
        
        p = lib.find_best_neighbor(self._cuda_frame, x, z, radius)
        lib_res = p.contents
        if lib_res[2] == 1:
            res = (lib_res[0], lib_res[1])
        else:
            res = None
            
        lib.free_waypoint(p)
        return res
    
    def count(self) -> int:
        return lib.count(self._cuda_frame)
    
    def check_in_graph(self, x: int, z: int) -> bool:
        return lib.check_in_graph(self._cuda_frame, x, z)
    
