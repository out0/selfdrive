import ctypes
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from model.waypoint import Waypoint
import math
from enum import Enum

LIBNAME = "/usr/local/lib/libdriveless-cudac.so"
lib = ctypes.CDLL(LIBNAME)

lib.load_frame.restype = ctypes.c_void_p
lib.load_frame.argtypes = [
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes.c_int, # width
    ctypes.c_int, # height
    ctypes.c_int, # min_dist_x
    ctypes.c_int, # min_dist_z
    ctypes.c_int, # lower_bound_x
    ctypes.c_int, # lower_bound_z
    ctypes.c_int, # upper_bound_x
    ctypes.c_int  # upper_bound_z
]

lib.destroy_frame.restype = None
lib.destroy_frame.argtypes = [ctypes.c_void_p]


lib.set_goal.restype = ctypes.c_bool
lib.set_goal.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # goalX
    ctypes.c_int, # goalZ
]

lib.set_goal_vectorized.restype = ctypes.c_bool
lib.set_goal_vectorized.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # goalX
    ctypes.c_int, # goalZ
]

lib.copy_back.restype = None
lib.copy_back.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1)
]

lib.get_color_frame.restype = None
lib.get_color_frame.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(dtype=ctypes.c_uint8, ndim=1)
]

lib.compute_feasible_path.restype = None
lib.compute_feasible_path.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
    ctypes.c_int, # count
    ctypes.c_bool
]

lib.best_waypoint_for_heading.restype = ctypes.POINTER(ctypes.c_int * 2)
lib.best_waypoint_for_heading.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # goal_x
    ctypes.c_int, # goal_z
    ctypes.c_float # heading
]

lib.best_waypoint.restype = ctypes.POINTER(ctypes.c_int * 2)
lib.best_waypoint.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_int, # goal_x
    ctypes.c_int # goal_z
]

lib.free_waypoint.restype = None
lib.free_waypoint.argtypes = [
    ctypes.POINTER(ctypes.c_int * 2)
]


lib.get_class_cost.restype = ctypes.c_int
lib.get_class_cost.argtypes = [ctypes.c_int]


class GridDirection (Enum):
    ALL = 0xff
    HEADING_0 = 0x01
    HEADING_22_5 = 0x02
    HEADING_45 = 0x04
    HEADING_67_5 = 0x08
    HEADING_90 = 0x10
    HEADING_MINUS_22_5 = 0x20
    HEADING_MINUS_45 = 0x40
    HEADING_MINUS_67_5 = 0x80


class CudaFrame:
    _cpu_frame: np.ndarray
    _cuda_frame: ctypes.c_void_p
    _orig_shape: tuple[int, int, int]
    _flatten_size: int
    _min_dist_x: int
    _min_dist_z: int
    _lower_bound: Waypoint
    _upper_bound: Waypoint
    
    def __get_flatten_size(self, frame: np.ndarray) -> int:
        size = 1
        for i in range(len(frame.shape)):
            size = size * frame.shape[i]
        return size
    
    def __flatten(self) -> None:
        self._cpu_frame = self._cpu_frame.reshape(self._flatten_size)
    
    def __unflatten(self) -> None:
        self._cpu_frame = self._cpu_frame.reshape(self._orig_shape)
    
    def __init__ (self, frame: np.ndarray, min_dist_x: int,  min_dist_z: int, lower_bound: Waypoint, upper_bound: Waypoint) -> None:
        self._cuda_frame = None
        self._cpu_frame = np.ascontiguousarray(frame, dtype=np.float32)
        self._flatten_size = self.__get_flatten_size(self._cpu_frame)
        self._orig_shape = (frame.shape[0], frame.shape[1], frame.shape[2])

        self._min_dist_x = min_dist_x
        self._min_dist_z = min_dist_z
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        
        self.__flatten()
        self._cuda_frame = lib.load_frame(
            self._cpu_frame,
            self._orig_shape[1], 
            self._orig_shape[0],
            min_dist_x,
            min_dist_z,
            lower_bound.x,
            lower_bound.z,
            upper_bound.x,
            upper_bound.z
        )
        self.__unflatten()
        
        
    def __del__(self):
        if self._cuda_frame is None:
            return
        lib.destroy_frame(self._cuda_frame)
        
    
    def set_goal(self, goal: Waypoint) -> np.ndarray:
        self.__flatten()
        lib.set_goal(self._cuda_frame, goal.x, goal.z)
        lib.copy_back(self._cuda_frame, self._cpu_frame)
        self.__unflatten()
        
    def set_goal_vectorized(self, goal: Waypoint) -> np.ndarray:
        self.__flatten()
        lib.set_goal_vectorized(self._cuda_frame, goal.x, goal.z)
        lib.copy_back(self._cuda_frame, self._cpu_frame)
        self.__unflatten()
        
    def get_frame (self) -> np.ndarray:
        return self._cpu_frame
    
    def get_shape(self) -> tuple[int, int, int]:
        return self._orig_shape
    
    def get_color_frame(self) -> np.ndarray:
        color_f = np.zeros(self._flatten_size, dtype=np.uint8)
        lib.get_color_frame(self._cuda_frame, color_f)
        return color_f.reshape(self._orig_shape)
    
    def compute_feasible_path(self, path: list[Waypoint], compute_heading: bool = False ) -> np.ndarray:
        count = len(path)
        wp = np.ndarray((count, 4), dtype=np.float32)
        res = np.ndarray(count, dtype=np.int8)
        i = 0
        for p in path:
            wp[i, 0] = p.x
            wp[i, 1] = p.z
            wp[i, 2] = p.heading
            wp[i, 3] = 0
            i += 1
        
        if (count > 0):
            lib.compute_feasible_path(self._cuda_frame, wp.reshape(4*count), count, compute_heading)
        
        i = 0
        for p in wp:
            res[i] = round(wp[i, 3])
            i += 1

        if compute_heading:
            i = 0
            for p in path:
                p.heading = wp[i,2]
                i += 1
            
        return res
    

    def check_waypoint_feasible(self, p: Waypoint) -> bool:
        r = math.radians(p.heading)
        c = math.cos(r)
        s = math.sin(r)

        width = self._orig_shape[1]
        height = self._orig_shape[0]
        
        min_dist_x = round(self._min_dist_x / 2)
        min_dist_z = round(self._min_dist_z / 2)

        for z in range (-min_dist_z, min_dist_z + 1):
            for x in range (-min_dist_x, min_dist_x + 1):

                xl = round(x * c - z * s) + p.x
                zl = round(x * s + z * c) + p.z

                if (xl < 0 or xl >= width):
                    continue

                if (zl < 0 or zl >= height):
                    continue

                if (xl >= self._lower_bound.x and xl <= self._upper_bound.x) and\
                    (zl >= self._upper_bound.z and zl <= self._lower_bound.z):
                    continue

                segmentation_class = round(self._cpu_frame[zl, xl, 0])

                if lib.get_class_cost(segmentation_class) < 0:
                    return False
        
        return True
    
    def find_best_cost_waypoint_with_heading(self, goal_x: int, goal_z: int, heading: float) -> Waypoint:
        
        p = lib.best_waypoint_for_heading(self._cuda_frame, goal_x, goal_z, heading)
        waypoint = p.contents
        lib.free_waypoint(p)
            
        return Waypoint(waypoint[0], waypoint[1], heading)
    
    def find_best_cost_waypoint(self, goal_x: int, goal_z: int) -> Waypoint:
        
        p = lib.best_waypoint(self._cuda_frame, goal_x, goal_z)   
        waypoint = p.contents
        x = waypoint[0]
        z = waypoint[1]
        lib.free_waypoint(p)
        
        heading = Waypoint.compute_heading(Waypoint(x, z, 0), Waypoint(goal_x, goal_z, 0))
        
        return Waypoint(x, z, heading)