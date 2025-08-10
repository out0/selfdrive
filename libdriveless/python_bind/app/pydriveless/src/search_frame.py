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
        


class SearchFrame:
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

        SearchFrame.setup_cpp_lib()
        self._cuda_ptr = SearchFrame.lib.search_frame_initialize(
            width, 
            height,
            lower_bound[0], 
            lower_bound[1], 
            upper_bound[0], 
            upper_bound[1])

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(SearchFrame, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
            
        SearchFrame.lib = ctypes.CDLL(lib_path)

        SearchFrame.lib.search_frame_initialize.restype = ctypes.c_void_p
        SearchFrame.lib.search_frame_initialize.argtypes = [
            ctypes.c_int, # width
            ctypes.c_int, # height
            ctypes.c_int, # lowerBoundX
            ctypes.c_int, # lowerBoundZ
            ctypes.c_int, # upperBoundX
            ctypes.c_int, # upperBoundZ
        ]
        
        SearchFrame.lib.search_frame_destroy.restype = None
        SearchFrame.lib.search_frame_destroy.argtypes = [
            ctypes.c_void_p, # self
        ]
        
        SearchFrame.lib.search_frame_copy_data.restype = None
        SearchFrame.lib.search_frame_copy_data.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ]    
        
        SearchFrame.lib.search_frame_copy_back.restype = None
        SearchFrame.lib.search_frame_copy_back.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
        ] 
        
        SearchFrame.lib.export_to_color_frame.restype = None
        SearchFrame.lib.export_to_color_frame.argtypes = [
            ctypes.c_void_p, # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_uint8, ndim=1)
        ]
        
        SearchFrame.lib.set_class_colors.restype = None
        SearchFrame.lib.set_class_colors.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #numColors
            np.ctypeslib.ndpointer(dtype=ctypes.c_uint, ndim=1)
        ]
        
        SearchFrame.lib.set_class_costs.restype = None
        SearchFrame.lib.set_class_costs.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #numClasses
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1)
        ]
        
        
        SearchFrame.lib.get_class_cost.restype = ctypes.c_float
        SearchFrame.lib.get_class_cost.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   #classId
        ]

        SearchFrame.lib.get_cost.restype = ctypes.c_double
        SearchFrame.lib.get_cost.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrame.lib.get_traversability.restype = ctypes.c_int
        SearchFrame.lib.get_traversability.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrame.lib.is_traversable.restype = ctypes.c_bool
        SearchFrame.lib.is_traversable.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        
        SearchFrame.lib.is_traversable_on_angle.restype = ctypes.c_bool
        SearchFrame.lib.is_traversable_on_angle.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            ctypes.c_float,                 # angle_rad
            ctypes.c_bool                   # precision_check
        ]
        
        SearchFrame.lib.process_safe_distance_zone.restype = None
        SearchFrame.lib.process_safe_distance_zone.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_bool,                  # compute_vectorized
            ctypes.c_int,                   # min_distance_x
            ctypes.c_int                    # min_distance_z
        ]
        
        SearchFrame.lib.check_feasible_path.restype = ctypes.c_bool
        SearchFrame.lib.check_feasible_path.argtypes = [
            ctypes.c_void_p,                # self
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), #path
            ctypes.c_int,                   # count
            ctypes.c_int,                   # minDistX
            ctypes.c_int,                   # minDistZ
            ctypes.c_bool                   # copyback information on individual waypoint check 
        ]        
        SearchFrame.lib.read_cell.restype = None
        SearchFrame.lib.read_cell.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1), # return data
        ]
        
        SearchFrame.lib.write_cell.restype = None
        SearchFrame.lib.write_cell.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
            ctypes.c_float,                 # val_1
            ctypes.c_float,                 # val_2
            ctypes.c_float,                 # val_3
        ]

        SearchFrame.lib.is_obstacle.restype = ctypes.c_bool
        SearchFrame.lib.is_obstacle.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        

        SearchFrame.lib.process_distance_to_goal.restype = None
        SearchFrame.lib.process_distance_to_goal.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]
        SearchFrame.lib.get_distance_to_goal.restype = ctypes.c_float
        SearchFrame.lib.get_distance_to_goal.argtypes = [
            ctypes.c_void_p,                # self
            ctypes.c_int,                   # x
            ctypes.c_int,                   # z
        ]


    def __getitem__(self, key) -> float3:
        x, z = key
        cell_data = np.zeros(3, dtype=np.float32)
        SearchFrame.lib.read_cell(self._cuda_ptr, x, z, cell_data)
        return float3(cell_data[0], cell_data[1], cell_data[2])
    
    def __setitem__(self, key, val: float3) -> None:
        x, z = key
        SearchFrame.lib.write_cell(self._cuda_ptr, x, z, val.x, val.y, val.z)

    
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
        size = self.__get_flatten_size(frame)
        #orig_shape = (frame.shape[0], frame.shape[1], frame.shape[2])
        f = np.ascontiguousarray(frame.reshape(size), dtype=np.float32)
        SearchFrame.lib.search_frame_copy_data(self._cuda_ptr, f)
        #frame.reshape(orig_shape)
        
    def get_color_frame(self) -> np.ndarray:
        color_frame = np.zeros((3 * self.__height * self.__width), dtype=np.uint8)
        SearchFrame.lib.export_to_color_frame(self._cuda_ptr, color_frame)
        return color_frame.reshape((self.__height, self.__width, 3))
    
    def set_class_colors(self, colors: np.ndarray):
        numClasses = colors.shape[0]        
        f = np.ascontiguousarray(colors.reshape(numClasses * 3), dtype=np.uint32)
        SearchFrame.lib.set_class_colors(self._cuda_ptr, numClasses, f)
        #colors.reshape((numColors, 3))
        
    def set_class_costs(self, costs: np.ndarray) -> None:
        numClasses = costs.shape[0]
        f = np.ascontiguousarray(costs.reshape(numClasses), dtype=np.float32)
        SearchFrame.lib.set_class_costs(self._cuda_ptr, numClasses, f)
    
    def get_class_cost(self, class_id: int) -> float:
        return SearchFrame.lib.get_class_cost(self._cuda_ptr, class_id)
    
    def get_cost(self, x: int, z: int) -> float:
        return SearchFrame.lib.get_cost(self._cuda_ptr, x, z)

    def get_traversability(self, x: int, z: int) -> int:
        return SearchFrame.lib.get_traversability(self._cuda_ptr, x, z)
    
    def is_traversable(self, x: int, z: int, heading: angle = None, precision_check: bool = False) -> bool:
        if heading is not None:
            return SearchFrame.lib.is_traversable_on_angle(self._cuda_ptr, x, z, heading.rad(), precision_check)
        return SearchFrame.lib.is_traversable(self._cuda_ptr, x, z)
    
    def is_obstacle(self, x: int, z: int) -> bool:
        return SearchFrame.lib.is_obstacle(self._cuda_ptr, x, z)

    def process_safe_distance_zone(self, min_distance: tuple[int, int], compute_vectorized: bool) -> float:
        self.__copy_back_frame = None
        self._last_min_dist = min_distance
        return SearchFrame.lib.process_safe_distance_zone(self._cuda_ptr, compute_vectorized, min_distance[0], min_distance[1])

    def process_distance_to_goal(self, x: int, z: int) -> None:
        self.__copy_back_frame = None
        SearchFrame.lib.process_distance_to_goal(self._cuda_ptr, x, z)

    def get_distance_to_goal(self, x: int, z: int) -> float:
        return SearchFrame.lib.get_distance_to_goal(self._cuda_ptr, x, z)

    def get_last_min_distance(self) -> tuple[int, int]:
        if not hasattr(self, "_last_min_dist"):
            raise Exception("No last min distance set")
        return self._last_min_dist

    def get_frame(self) -> np.ndarray:
        if self.__copy_back_frame is None:
            frame = np.zeros((self.__height, self.__width, 3), dtype=np.float32)
            size = self.__get_flatten_size(frame)
            f = np.ascontiguousarray(frame.reshape(size), dtype=np.float32)
            SearchFrame.lib.search_frame_copy_back(self._cuda_ptr, f)
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
        
        res = SearchFrame.lib.check_feasible_path(
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