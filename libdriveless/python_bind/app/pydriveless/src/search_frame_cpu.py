import math
import ctypes
import numpy as np
from .waypoint import Waypoint
from .angle import angle
import os

class float3:
    x: float
    y: float
    z: float
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.x}, {self.y}, {self.z}"
        


class SearchFrameCPU:
    __width: int
    __height: int
    __lower_bound: tuple[int, int]
    __upper_bound: tuple[int, int]
    __copy_back_frame: np.ndarray
      
    
    def __init__(self, 
                 width: int, 
                 height: int,
                 lower_bound: tuple[int, int],
                 upper_bound: tuple[int, int]):
        
        if width == 0 or height == 0:
            raise Exception("cant create a search frame with these dimensions: {self.__width} x {self.__height}")
        
        self.__width = width
        self.__height = height
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__copy_back_frame = None

        SearchFrameCPU.setup_cpp_lib()
        self._cuda_ptr = SearchFrameCPU.lib.search_frame_initialize(
            width, 
            height,
            lower_bound[0], 
            lower_bound[1], 
            upper_bound[0], 
            upper_bound[1])

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(SearchFrameCPU, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
            
        SearchFrameCPU.lib = ctypes.CDLL(lib_path)

        SearchFrameCPU.lib.search_frame_initialize_cpu.restype = ctypes.c_void_p
        SearchFrameCPU.lib.search_frame_initialize_cpu.argtypes = [
            ctypes.c_int, # width
            ctypes.c_int, # height
            ctypes.c_int, # lowerBoundX
            ctypes.c_int, # lowerBoundZ
            ctypes.c_int, # upperBoundX
            ctypes.c_int, # upperBoundZ
        ]
        
        SearchFrameCPU.lib.search_frame_destroy_cpu.restype = None
        SearchFrameCPU.lib.search_frame_destroy_cpu.argtypes = [
            ctypes.c_void_p, # self
        ]
        
        SearchFrameCPU.lib.search_frame_copy_data_cpu.restype = None
        SearchFrameCPU.lib.search_frame_copy_data_cpu.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ]    
        
        SearchFrameCPU.lib.search_frame_copy_back_cpu.restype = None
        SearchFrameCPU.lib.search_frame_copy_back_cpu.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ] 
        
        SearchFrameCPU.lib.export_to_color_frame_cpu.restype = None
        SearchFrameCPU.lib.export_to_color_frame_cpu.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_uint8, ndim=1)
        ]
        
        SearchFrameCPU.lib.set_class_colors_cpu.restype = None
        SearchFrameCPU.lib.set_class_colors_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #numColors
            np.ctypeslib.ndpointer(dtype=ctypes.c_uint, ndim=1)
        ]
        
        SearchFrameCPU.lib.set_class_costs_cpu.restype = None
        SearchFrameCPU.lib.set_class_costs_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #numClasses
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1)
        ]
        
        
        SearchFrameCPU.lib.get_class_cost_cpu.restype = ctypes.c_float
        SearchFrameCPU.lib.get_class_cost_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #classId
        ]

        SearchFrameCPU.lib.get_cost_cpu.restype = ctypes.c_double
        SearchFrameCPU.lib.get_cost_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrameCPU.lib.get_traversability_cpu.restype = ctypes.c_int
        SearchFrameCPU.lib.get_traversability_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrameCPU.lib.is_traversable_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_traversable_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrameCPU.lib.is_traversable_on_angle_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_traversable_on_angle_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            ctypes.c_float,                 # angle_rad
            ctypes.c_bool                   # precision_check
        ]
        
        SearchFrameCPU.lib.process_safe_distance_zone_cpu.restype = None
        SearchFrameCPU.lib.process_safe_distance_zone_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_bool,                  # compute_vectorized
            ctypes.c_int,                   # min_distance_x
            ctypes.c_int                    # min_distance_z
        ]
        
        SearchFrameCPU.lib.check_feasible_path_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.check_feasible_path_cpu.argtypes = [
            ctypes.c_void_p,                # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), #path
            ctypes.c_int,                   # count
            ctypes.c_int,                   # minDistX
            ctypes.c_int,                   # minDistZ
            ctypes.c_bool                   # copyback information on individual waypoint check 
        ]        
        SearchFrameCPU.lib.read_cell_cpu.restype = None
        SearchFrameCPU.lib.read_cell_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), # return data
        ]
        
        SearchFrameCPU.lib.write_cell_cpu.restype = None
        SearchFrameCPU.lib.write_cell_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            ctypes.c_float,                 # val_1
            ctypes.c_float,                 # val_2
            ctypes.c_float,                 # val_3
        ]

        SearchFrameCPU.lib.is_obstacle_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_obstacle_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        

        SearchFrameCPU.lib.process_distance_to_goal_cpu.restype = None
        SearchFrameCPU.lib.process_distance_to_goal_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        SearchFrameCPU.lib.get_distance_to_goal_cpu.restype = ctypes.c_float
        SearchFrameCPU.lib.get_distance_to_goal_cpu.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]

        SearchFrameCPU.lib.is_safe_zone_checked_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_safe_zone_checked_cpu.argtypes = [
            ctypes.c_void_p,                # self
        ]
        
        SearchFrameCPU.lib.is_vectorial_safe_zone_checked_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_vectorial_safe_zone_checked_cpu.argtypes = [
            ctypes.c_void_p,                # self
        ]

        SearchFrameCPU.lib.is_distance_to_goal_processed_cpu.restype = ctypes.c_bool
        SearchFrameCPU.lib.is_distance_to_goal_processed_cpu.argtypes = [
            ctypes.c_void_p,                # self
        ]        



    def __getitem__(self, key) -> float3:
        x, z = key
        cell_data = np.zeros(3, dtype=np.float32)
        SearchFrameCPU.lib.read_cell(self._cuda_ptr, x, z, cell_data)
        return float3(cell_data[0], cell_data[1], cell_data[2])
    
    def __setitem__(self, key, val: float3) -> None:
        x, z = key
        SearchFrameCPU.lib.write_cell(self._cuda_ptr, x, z, val.x, val.y, val.z)

    
    def get_cuda_ptr(self) -> ctypes.c_void_p:
        if not hasattr(self, "_cuda_ptr"):
            raise Exception("SearchFrame not initialized")
        return self._cuda_ptr

    def width(self) -> int:
        return self.__width
    
    def height(self) -> int:
        return self.__height
    
    def lowerBound(self) -> tuple[int, int]:
        return self.__lower_bound

    def upperBound(self) -> tuple[int, int]:
        return self.__upper_bound
    
    def __get_flatten_size(self, frame: np.ndarray) -> int:
        size = 1
        for i in range(len(frame.shape)):
            size = size * frame.shape[i]
        return size
    
    def set_frame_data(self, frame: np.ndarray):
        self.__copy_back_frame = None
        if (len(frame.shape) < 3):
            raise Exception(f"frame shape does not match search frame dimensions: {frame.shape} vs {self.__height} x {self.__width} x 3")
        elif frame.shape[0] != self.__height or frame.shape[1] != self.__width or frame.shape[2] != 3:
            raise Exception(f"frame shape does not match search frame dimensions: {frame.shape} vs {self.__height} x {self.__width} x 3")
        size = self.__get_flatten_size(frame)
        #orig_shape = (frame.shape[0], frame.shape[1], frame.shape[2])
        f = np.ascontiguousarray(frame.reshape(size), dtype=np.float32)
        SearchFrameCPU.lib.search_frame_copy_data(self._cuda_ptr, f)
        #frame.reshape(orig_shape)
        
    def get_color_frame(self) -> np.ndarray:
        color_frame = np.zeros((3 * self.__height * self.__width), dtype=np.uint8)
        SearchFrameCPU.lib.export_to_color_frame(self._cuda_ptr, color_frame)
        return color_frame.reshape((self.__height, self.__width, 3))
    
    def set_class_colors(self, colors: np.ndarray):
        numClasses = colors.shape[0]        
        f = np.ascontiguousarray(colors.reshape(numClasses * 3), dtype=np.uint32)
        SearchFrameCPU.lib.set_class_colors(self._cuda_ptr, numClasses, f)
        #colors.reshape((numColors, 3))
        
    def set_class_costs(self, costs: np.ndarray) -> None:
        numClasses = costs.shape[0]
        f = np.ascontiguousarray(costs.reshape(numClasses), dtype=np.float32)
        SearchFrameCPU.lib.set_class_costs(self._cuda_ptr, numClasses, f)
    
    def get_class_cost(self, class_id: int) -> float:
        return SearchFrameCPU.lib.get_class_cost(self._cuda_ptr, class_id)
    
    def get_cost(self, x: int, z: int) -> float:
        return SearchFrameCPU.lib.get_cost(self._cuda_ptr, x, z)

    def get_traversability(self, x: int, z: int) -> int:
        return SearchFrameCPU.lib.get_traversability(self._cuda_ptr, x, z)
    
    def is_traversable(self, x: int, z: int, heading: angle = None, precision_check: bool = False) -> bool:
        if heading is not None:
            return SearchFrameCPU.lib.is_traversable_on_angle(self._cuda_ptr, x, z, heading.rad(), precision_check)
        return SearchFrameCPU.lib.is_traversable(self._cuda_ptr, x, z)
    
    def is_obstacle(self, x: int, z: int) -> bool:
        return SearchFrameCPU.lib.is_obstacle(self._cuda_ptr, x, z)

    def process_safe_distance_zone(self, min_distance: tuple[int, int], compute_vectorized: bool) -> float:
        self.__copy_back_frame = None
        self._last_min_dist = min_distance
        return SearchFrameCPU.lib.process_safe_distance_zone(self._cuda_ptr, compute_vectorized, min_distance[0], min_distance[1])

    def process_distance_to_goal(self, x: int, z: int) -> None:
        self.__copy_back_frame = None
        SearchFrameCPU.lib.process_distance_to_goal(self._cuda_ptr, x, z)

    def get_distance_to_goal(self, x: int, z: int) -> float:
        return SearchFrameCPU.lib.get_distance_to_goal(self._cuda_ptr, x, z)

    def get_last_min_distance(self) -> tuple[int, int]:
        if not hasattr(self, "_last_min_dist"):
            raise Exception("No last min distance set")
        return self._last_min_dist

    def get_frame(self) -> np.ndarray:
        if self.__copy_back_frame is None:
            frame = np.zeros((self.__height, self.__width, 3), dtype=np.float32)
            size = self.__get_flatten_size(frame)
            f = np.ascontiguousarray(frame.reshape(size), dtype=np.float32)
            SearchFrameCPU.lib.search_frame_copy_back(self._cuda_ptr, f)
            self.__copy_back_frame = f.reshape((self.__height, self.__width, 3))
        return self.__copy_back_frame

    def check_feasible_path(self, min_distance: tuple[int, int], path: list[Waypoint], individual_waypoint_check: bool = False) -> bool:
        size = len(path)
        points = np.zeros((size, 4), dtype=np.float32)
        for i in range(len(path)):
            points[i, 0] = path[i].x
            points[i, 1] = path[i].z
            points[i, 2] = path[i].heading.rad()
            points[i, 3] = 0
            
        f = np.ascontiguousarray(points.reshape(4*size), dtype=np.float32)
        
        res = SearchFrameCPU.lib.check_feasible_path(
            self._cuda_ptr,
            f,
            size,
            min_distance[0],
            min_distance[1],
            individual_waypoint_check
        )

        if individual_waypoint_check:
            for i in range(len(path)):
                path[i].set_checked_as_feasible(points[i, 3] == 1.0)

        return res

    def lower_bound(self) -> tuple[int, int]:
        return self.__lower_bound
    
    def upper_bound(self) -> tuple[int, int]:
        return self.__upper_bound
    

    def is_safe_zone_checked (self) -> bool:
        return SearchFrameCPU.lib.is_safe_zone_checked(self._cuda_ptr)
    
    def is_vectorial_safe_zone_checked (self) -> bool:
        return SearchFrameCPU.lib.is_vectorial_safe_zone_checked(self._cuda_ptr)

    def is_distance_to_goal_processed (self) -> bool:
        return SearchFrameCPU.lib.is_distance_to_goal_processed(self._cuda_ptr)
