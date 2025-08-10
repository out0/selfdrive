import math
import ctypes
import numpy as np
from .angle import angle
import os


class Waypoint:
    __x: int
    __z: int
    __heading: angle
    __checked_as_feasible: bool
    
    def __init__(self, 
                 x: int, 
                 z: int, 
                 heading: angle = None):
        self.__x = int(x)
        self.__z = int(z)
        self.__heading = heading
        
        if self.__heading is None:
            self.__heading = angle.new_rad(0)
    
        self.__checked_as_feasible = False
        Waypoint.setup_cpp_lib()

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(Waypoint, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
            
        Waypoint.lib = ctypes.CDLL(lib_path)

        Waypoint.lib.waypoint_distance_between.restype = ctypes.c_double
        Waypoint.lib.waypoint_distance_between.argtypes = [
            ctypes.c_int, # x1
            ctypes.c_int, # z1
            ctypes.c_int, # x2
            ctypes.c_int, # z2
        ]
        Waypoint.lib.waypoint_compute_heading.restype = ctypes.c_double
        Waypoint.lib.waypoint_compute_heading.argtypes = [
            ctypes.c_int, # x1
            ctypes.c_int, # z1
            ctypes.c_int, # x2
            ctypes.c_int, # z2
        ]
        Waypoint.lib.waypoint_distance_to_line.restype = ctypes.c_double
        Waypoint.lib.waypoint_distance_to_line.argtypes = [
            ctypes.c_int, # x1
            ctypes.c_int, # z1
            ctypes.c_int, # x2
            ctypes.c_int, # z2
            ctypes.c_int, # x_target
            ctypes.c_int, # z_target
        ]


    @property
    def x(self) -> int:
        return self.__x

    @property
    def z(self) -> int:
        return self.__z

    @property
    def heading(self) -> angle:
        return self.__heading

    def __str__(self):
        return f"({self.__x}, {self.__z}, {self.__heading.deg()})"
    
    
    def __eq__(self, other: 'Waypoint'):
        if other is None: return False
        return  self.__x == other.x and \
                self.__z == other.z and \
                self.__heading == other.heading
    
    def from_str(payload: str) -> 'Waypoint':
        if payload == 'None':
            return None
        
        payload = payload.replace("(","").replace(")", "")
        p = payload.split(",")
        return Waypoint(
            int(p[0]),
            int(p[1]),
            angle.new_deg(float(p[2]))
        )
       
    def distance_between(p1: 'Waypoint', p2: 'Waypoint') -> float:
        return Waypoint.lib.waypoint_distance_between(p1.__x, p1.__z, p2.__x, p2.__z)

    def compute_heading(p1: 'Waypoint', p2: 'Waypoint') -> angle:
        angle_rad = Waypoint.lib.waypoint_compute_heading(p1.__x, p1.__z, p2.__x, p2.__z)
        return angle.new_rad(angle_rad)
    
    @classmethod
    def distance_to_line(cls, line_p1: 'Waypoint', line_p2: 'Waypoint', p: 'Waypoint') -> float:
        return Waypoint.lib.waypoint_distance_to_line(line_p1.__x, line_p1.__z, line_p2.__x, line_p2.__z, p.__x, p.__z)
        
    
    @classmethod
    def mid_point(cls, p1: 'Waypoint', p2: 'Waypoint') -> 'Waypoint':
        return Waypoint(math.floor((p2.__x + p1.__x)/2),  math.floor((p2.__z + p1.__z)/2))
    
    @classmethod
    def clip(cls, p: 'Waypoint', width: int, height: int) -> 'Waypoint':
        res = Waypoint(p.__x, p.__z, p.__heading)
        
        if res.x < 0:
            res.x = 0
        if res.x >= width:
            res.x = width - 1
        if res.z < 0:
            res.z = 0
        if res.z >= height:
            res.z = height - 1            
        return res

    def clone(self) -> 'Waypoint':
        return Waypoint(
            self.__x,
            self.__z,
            self.__heading
        )
        
    def set_checked_as_feasible(self, val: bool) -> None:
        self.__checked_as_feasible = val

    def is_checked_as_feasible(self) -> bool:
        return self.__checked_as_feasible