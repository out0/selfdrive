import ctypes
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from model.waypoint import Waypoint
import math
from enum import Enum
from utils.cudac.cuda_frame import CudaFrame

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


lib.find_nearest_neighbor.restype = ctypes.POINTER(ctypes.c_int * 3)
lib.find_nearest_neighbor.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int # Z
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


lib.get_parent.restype = ctypes.POINTER(ctypes.c_int * 3)
lib.get_parent.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int # Z    
]

lib.link.restype = None
lib.link.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_void_p, 
    ctypes.c_int, # parent_x
    ctypes.c_int, # parent_z
    ctypes.c_int, # X
    ctypes.c_int # Z    
]

lib.list_nodes.restype = None
lib.list_nodes.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1)
]

lib.get_cost.restype = ctypes.c_float
lib.get_cost.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # X
    ctypes.c_int # Z
]

lib.optimize_graph.restype = None
lib.optimize_graph.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int, # X
    ctypes.c_int, # Z
    ctypes.c_int, # parent_x
    ctypes.c_int, # parent_z
    ctypes.c_float, # cost
    ctypes.c_float # search_radius
]



class CudaGraph:
    _cuda_graph: ctypes.c_void_p
   
    def __init__ (self, 
                width: int, 
                height: int) -> None:
        self._cuda_graph = None
        self._cuda_graph = lib.load_frame(
            width,
            height)
        
    def __del__(self):
        if self._cuda_graph is None:
            return
        lib.destroy_frame(self._cuda_graph)
        
    def clear(self):
        if self._cuda_graph is None:
            return
        lib.clear(self._cuda_graph)

    def add_point(self, x: int, z: int, parent_x: int, parent_z: int, cost: float):
        if self._cuda_graph is None:
            return
        
        lib.add_point(self._cuda_graph, x, z, parent_x, parent_z, cost)


    def find_best_neighbor(self, x: int, z: int, radius: float) -> tuple[int, int]:
        if self._cuda_graph is None:
            return
        
        p = lib.find_best_neighbor(self._cuda_graph, x, z, radius)
        lib_res = p.contents
        if lib_res[2] == 1:
            res = (lib_res[0], lib_res[1])
        else:
            res = None
            
        lib.free_waypoint(p)
        return res

    def find_nearest_neighbor(self, x: int, z: int) -> tuple[int, int]:
        if self._cuda_graph is None:
            return
        
        p = lib.find_nearest_neighbor(self._cuda_graph, x, z)
        lib_res = p.contents
        if lib_res[2] == 1:
            res = (lib_res[0], lib_res[1])
        else:
            res = None
            
        lib.free_waypoint(p)
        return res
    
    
    def count(self) -> int:
        return lib.count(self._cuda_graph)
    
    def check_in_graph(self, x: int, z: int) -> bool:
        return lib.check_in_graph(self._cuda_graph, x, z)
    
    def get_parent(self, x: int, z: int) -> tuple[int, int]:
        if self._cuda_graph is None:
            return
        
        p = lib.get_parent(self._cuda_graph, x, z)
        lib_res = p.contents
        if lib_res[2] == 1:
            res = (lib_res[0], lib_res[1])
        else:
            res = None
            
        lib.free_waypoint(p)
        return res

    def list_nodes(self) -> np.ndarray:
        count = self.count()
        shape = (count, 5)
        res = np.zeros(shape, dtype=np.float32)
        f = res.reshape(count * 5)
        lib.list_nodes(self._cuda_graph, f, count)
        res.reshape(shape)
        return res
        
    def get_cost(self, x: int, z: int) -> bool:
        return lib.get_cost(self._cuda_graph, x, z)
    
    
    def optimize_graph(self, x: int, z: int, parent_x: int, parent_z: int, cost: float, search_radius: float):
        lib.optimize_graph(self._cuda_graph, x, z, parent_x, parent_z, cost, search_radius)