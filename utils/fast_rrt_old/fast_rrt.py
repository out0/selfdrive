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
from model.physical_parameters import PhysicalParameters

LIBNAME = "/usr/local/lib/libfast-rrts.so"
lib = ctypes.CDLL(LIBNAME)

FLOAT_SIZE_BYTES = 4  # sizeof_float

lib.init.restype = ctypes.c_void_p
lib.init.argtypes = [
    ctypes.c_int, # width
    ctypes.c_int, # height
    ctypes.c_float, #og_real_width_m
    ctypes.c_float, #og_real_height_m
    ctypes.c_int, # min_dist_x,
    ctypes.c_int, # min_dist_z,
    ctypes.c_int, # lower_bound_ego_x,
    ctypes.c_int, # lower_bound_ego_z,
    ctypes.c_int, # upper_bound_ego_x,
    ctypes.c_int, # upper_bound_ego_z
    ctypes.c_float # max_steering_angle
]

lib.destroy.restype = None
lib.destroy.argtypes =[ctypes.c_void_p]

lib.gen_path_waypoint.restype = ctypes.c_int
lib.gen_path_waypoint.argtypes = [
    ctypes.c_void_p, # self
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), # res
    ctypes.c_int, #start_x,
    ctypes.c_int, # start_z,
    ctypes.c_float, # start_heading,
    ctypes.c_float, # velocity_m_s,
    ctypes.c_float, #  sterr_angle,
    ctypes.c_float # path_size
]

lib.connect_nodes_with_path.restype = ctypes.c_void_p
lib.connect_nodes_with_path.argtypes = [
    ctypes.c_void_p, # self
    ctypes.c_int, #start_x,
    ctypes.c_int, # start_z,
    ctypes.c_float, # start_heading,
    ctypes.c_int, #end_x,
    ctypes.c_int, # end_z,
    ctypes.c_float # velocity_m_s,
]

lib.connect_nodes_with_path_free.restype = ctypes.c_void_p
lib.connect_nodes_with_path_free.argtypes = [
    ctypes.c_void_p, # float
]

lib.set_plan_data.restype = None
lib.set_plan_data.argtypes = [
    ctypes.c_void_p, # self
    ctypes.c_void_p, # cuda_frame
    ctypes.c_int, # start_x
    ctypes.c_int, # start_z
    ctypes.c_float, # start_heading
    ctypes.c_int, # goal_x
    ctypes.c_int, # goal_z
    ctypes.c_float, # goal_heading
    ctypes.c_float, # velocity_m_s
]
lib.search.restype = None
lib.search.argtypes = [
    ctypes.c_void_p, # self
]
lib.cancel.restype = None
lib.cancel.argtypes = [
    ctypes.c_void_p, # self
]
lib.is_planning.restype = ctypes.c_bool
lib.is_planning.argtypes = [
    ctypes.c_void_p, # self
]

lib.get_path_size.restype = ctypes.c_int
lib.get_path_size.argtypes = [
    ctypes.c_void_p, # self
]
lib.get_path.restype = None
lib.get_path.argtypes = [
    ctypes.c_void_p, # self
    np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
]


class FastRRT:
    __fast_rrt_cuda: ctypes.c_void_p
    
    def __init__ (self):
        self.__fast_rrt_cuda = None
        self.__fast_rrt_cuda = lib.init(
            PhysicalParameters.OG_WIDTH,
            PhysicalParameters.OG_HEIGHT,
            PhysicalParameters.OG_REAL_WIDTH,
            PhysicalParameters.OG_REAL_HEIGHT,
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND.x,
            PhysicalParameters.EGO_LOWER_BOUND.z,
            PhysicalParameters.EGO_UPPER_BOUND.x,
            PhysicalParameters.EGO_UPPER_BOUND.z,
            PhysicalParameters.MAX_STEERING_ANGLE)
        
    def __del__(self):
        if self.__fast_rrt_cuda is None:
            return
        lib.destroy(self.__fast_rrt_cuda)
        
    def gen_path_waypoint(self,
            start: Waypoint,
            velocity_m_s: float,
            steering_angle: float,
            path_size) -> list[Waypoint]:
        
        buffer_size = 3 * math.floor(path_size)
        res = np.ascontiguousarray(np.zeros((buffer_size)), dtype=np.float32)
        
        count = lib.gen_path_waypoint(
            self.__fast_rrt_cuda,
            res,
            start.x,
            start.z,
            start.heading,
            velocity_m_s,
            steering_angle,
            path_size
        )
        
        res = res.reshape((count, 3))
        
        list = []
                
        for i in range(res.shape[0]):
            list.append(Waypoint(
                x = math.floor(res[i, 0]),
                z = math.floor(res[i, 1]),
                heading= res[i, 2]
            ))
            
        return list
       
    def connect_nodes_with_path(self,
                            start: Waypoint,
                            end: Waypoint,
                            velocity_m_s: float) -> list[Waypoint]:
        
        points = lib.connect_nodes_with_path(
            self.__fast_rrt_cuda,
            start.x,
            start.z,
            start.heading,
            end.x,
            end.z,
            velocity_m_s)

        data = ctypes.cast(points, ctypes.POINTER(ctypes.c_float))

        size = math.floor(data[0])
        path = []

        for i in range(size):
            pos = 3*i + 1
            path.append(
                Waypoint(
                    x = math.floor(data[pos]),
                    z = math.floor(data[pos+1]),
                    heading = data[pos+2]
                )
            )
            
        lib.connect_nodes_with_path_free(points)

        return path
    
    def set_plan_data(self, og: CudaFrame, start: Waypoint, end: Waypoint, velocity_m_s: float) -> None:
        lib.set_plan_data(
            self.__fast_rrt_cuda,
            og.get_cuda_frame(),
            start.x,
            start.z,
            start.heading,
            end.x,
            end.z,
            end.heading,
            velocity_m_s
        )
    
    def search(self) -> None:
        lib.search(self.__fast_rrt_cuda)
    
    def cancel(self) -> None:
        lib.cancel(self.__fast_rrt_cuda)
    
    def is_planning(self) -> None:
        return lib.is_planning(self.__fast_rrt_cuda)

    def get_path(self) -> list[Waypoint]:
        if self.is_planning():
            return []
        
        path_size = lib.get_path_size(self.__fast_rrt_cuda)
        raw_path = np.ascontiguousarray(np.zeros((path_size * 3, 1), dtype=np.float32))
        lib.get_path(self.__fast_rrt_cuda, raw_path)
        
        raw_path = raw_path.reshape(path_size, 3)
        
        list = []
        
        for i in range(0, path_size):
            list.append(Waypoint(
                math.floor(raw_path[i, 0]),
                math.floor(raw_path[i, 1]),
                raw_path[i, 2],
            ))
        
        return list
        