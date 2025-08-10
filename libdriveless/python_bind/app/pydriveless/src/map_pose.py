import math, ctypes
from .angle import angle
import numpy as np
from enum import Enum
import os

class NearestPoseType(Enum):
    NotFound = 0,
    Before_P1 = 1,
    Exact_P1 = 2,
    After_P1 = 3,
    Before_P2 = 4,
    Exact_P2 = 5,
    After_P2 = 6


class NearestPoseSearchResult:
    list_pos: int
    pos_type: NearestPoseType
    best_segment_size: float
    best_distance_from_p1: float

    def __init__(self, list_pos: int,
                 pos_type: NearestPoseType,
                 best_segment_size: float,
                 best_distance_from_p1: float):
        self.list_pos = list_pos
        self.pos_type = pos_type
        self.best_segment_size = best_segment_size
        self.best_distance_from_p1 = best_distance_from_p1
        





class MapPose:
    __x: float
    __y: float
    __z: float
    __heading: angle

    def __init__(self, 
                 x: float, 
                 y: float, 
                 z: float, 
                 heading: angle = None):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__heading = heading
        
        if self.__heading is None:
            self.__heading = angle.new_rad(0)
            
        if (isinstance(heading, int) or isinstance(heading, float)):
            self.__heading = angle.new_rad(heading)
    
        MapPose.setup_cpp_lib()
    
    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(MapPose, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")

        MapPose.lib = ctypes.CDLL(lib_path)

        MapPose.lib.map_pose_distance_to_line_2d.restype = ctypes.c_double
        MapPose.lib.map_pose_distance_to_line_2d.argtypes = [
            ctypes.c_double, # x1
            ctypes.c_double, # y1
            ctypes.c_double, # x2
            ctypes.c_double, # y2
            ctypes.c_double, # x
            ctypes.c_double # y
        ]
        
        MapPose.lib.map_pose_compute_path_heading_2d.restype = ctypes.c_double
        MapPose.lib.map_pose_compute_path_heading_2d.argtypes = [
            ctypes.c_double, # x1
            ctypes.c_double, # y1
            ctypes.c_double, # x2
            ctypes.c_double # y2
        ]

        MapPose.lib.map_pose_project_on_path.restype = ctypes.c_void_p
        MapPose.lib.map_pose_project_on_path.argtypes = [
            ctypes.c_double, # x1
            ctypes.c_double, # y1
            ctypes.c_double, # x2
            ctypes.c_double, # y2
            ctypes.c_double, # x
            ctypes.c_double # y
        ]

        
        MapPose.lib.map_pose_project_on_path_free.restype = None
        MapPose.lib.map_pose_project_on_path_free.argtypes = [
            ctypes.c_void_p
        ]

        MapPose.lib.map_pose_free_pose_list.restype = None
        MapPose.lib.map_pose_free_pose_list.argtypes = [
            ctypes.c_void_p
        ]
        
        MapPose.lib.map_pose_store_pose_list.restype = ctypes.c_void_p
        MapPose.lib.map_pose_store_pose_list.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),
            ctypes.c_int # count
        ]  

        MapPose.lib.find_best_next_pose_on_list.restype = ctypes.c_int
        MapPose.lib.find_best_next_pose_on_list.argtypes = [
            ctypes.c_void_p, # list ptr
            ctypes.c_double, # location_x
            ctypes.c_double, # location_y
            ctypes.c_int, # first position on list to check
            ctypes.c_double, # min distance
            ctypes.c_int  # max_hopping
        ]

        
    
    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def z(self) -> float:
        return self.__z

    @property
    def heading(self) -> angle:
        return self.__heading

    
    def __eq__(self, other: 'MapPose'):
        return  self.__x == other.x and\
                self.__y == other.y and\
                self.__z == other.z and\
                self.__heading == other.heading
    
    def __add__(self, other) -> 'MapPose':
        if isinstance(other, MapPose):
            return MapPose (
                self.__x + other.x,
                self.__y + other.y,
                self.__z + other.z,
                self.__heading + other.heading
            )
        else:
            return MapPose (
                self.__x + other,
                self.__y + other,
                self.__z + other,
                self.__heading
            )
    def __sub__(self, other) -> 'MapPose':
        if isinstance(other, MapPose):
            return MapPose (
                self.__x - other.x,
                self.__y - other.y,
                self.__z - other.z,
                self.__heading - other.heading
            )
        else:
            return MapPose (
                self.__x - other,
                self.__y - other,
                self.__z - other,
                self.__heading
            )
            
    def clone(self) -> 'MapPose':
        return MapPose(
            self.__x,
            self.__y,
            self.__z,
            self.__heading
        )




    @classmethod    
    def are_close(cls, p1: 'MapPose', p2: 'MapPose') -> bool:
        return  math.isclose(p1.x, p2.x, rel_tol=1e-2) and\
                math.isclose(p1.y, p2.y, rel_tol=1e-2)
    
    @classmethod
    def distance_between(cls, p1 : 'MapPose', p2 : 'MapPose') -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx**2 + dy**2)
    
    @classmethod
    def dot(cls, p1 : 'MapPose', p2 : 'MapPose') -> float:
        return p1.x * p2.x + p1.y * p2.y



    @classmethod
    def distance_to_line(cls, line_p1: 'MapPose', line_p2: 'MapPose', p: 'MapPose') -> float:
        return MapPose.lib.map_pose_distance_to_line_2d(
            line_p1.x,
            line_p1.y,
            line_p2.x,
            line_p2.y,
            p.x,
            p.y
        )
    @classmethod
    def compute_path_heading(cls, p1: 'MapPose', p2: 'MapPose') -> angle:
        angle_rad = MapPose.lib.map_pose_compute_path_heading_2d(
            p1.x,
            p1.y,
            p2.x,
            p2.y
        )
        return angle.new_rad(angle_rad)


    # @classmethod
    # def project_on_path(cls, p1: 'MapPose', p2: 'MapPose', p: 'MapPose') -> tuple['MapPose', float, float]:
    #     ptr = MapPose.lib.map_pose_project_on_path(
    #         p1.x,
    #         p1.y,
    #         p2.x,
    #         p2.y,
    #         p.x,
    #         p.y
    #     )
        
    #     data = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
        
    #     pose = MapPose(data[0], data[1], data[2], angle.new_rad(data[3]))
    #     dist_p1 = data[4]
    #     path_size = data[5]
        
    #     MapPose.lib.map_pose_project_on_path_free(ptr)
        
    #     return pose, dist_p1, path_size

    @classmethod
    def dot_2D(cls, p1: 'MapPose', p2: 'MapPose'):
        return p1.x * p2.x + p1.y * p2.y

    @classmethod
    def project_on_path(cls, p1: 'MapPose', p2: 'MapPose', p: 'MapPose') -> tuple['MapPose', float, float]:
        path_size = MapPose.distance_between(p1, p2)

        if (path_size == 0):
            return None, 0, 0

        l = MapPose(
            x=((p2.x - p1.x) / path_size), 
            y=((p2.y - p1.y) / path_size), 
            z=0, 
            heading=angle.new_rad(0))
        
        v = MapPose(
            x=(p.x - p1.x),
            y=(p.y - p1.y),
            z=0, 
            heading=angle.new_rad(0))

        distance_from_p1 = MapPose.dot_2D(v, l)

        return (MapPose(
                x=p1.x + l.x * distance_from_p1,
                y=p1.y + l.y * distance_from_p1,
                z=0, 
                heading=angle.new_rad(0)),
                path_size,
                distance_from_p1)

    def __str__(self) -> str:
        return f"{self.__x}|{self.__y}|{self.__z}|{self.__heading.deg()}"

    @classmethod
    def from_str(cls, payload: str) -> 'MapPose':
        if payload == 'None':
            return None
        
        p = payload.split("|")
        return MapPose (
            float(p[0]),
            float(p[1]),
            float(p[2]),
            angle.new_deg(float(p[3]))
        )
    
    @classmethod
    def __squared_distance_to_midline(cls, p1: 'MapPose', p2: 'MapPose', p: 'MapPose'):
        dx = (0.5*(p2.x + p1.x))-p.x
        dy = (0.5*(p2.y + p1.y))-p.y
        return dx * dx + dy * dy

    @classmethod
    def __find_nearest_pose_in_list(cls, location: 'MapPose', poses: list['MapPose'], start: int, max_hopping: int) -> NearestPoseSearchResult:

        best_distance: float = 999999999
        best_pos: int = -1
        last_position_on_list: int = len(poses) - 1
        best_pos_type = NearestPoseType.NotFound
        best_segment_size: float = 0
        best_dist_from_p1: float = 0
        hopping: int = 0

        if start < 0:
            start = 0

        for i in range(start, last_position_on_list):
            if (max_hopping > 0 and hopping > max_hopping):
                break

            if MapPose.are_close(location, poses[i]):
                return NearestPoseSearchResult(i, NearestPoseType.Exact_P1, -1, -1)

            if MapPose.are_close(location, poses[i + 1]):
                return NearestPoseSearchResult(i + 1, NearestPoseType.Exact_P2, -1, -1)

            distance_to_segment = MapPose.__squared_distance_to_midline(poses[i], poses[i+1], location)

            if (distance_to_segment >= best_distance):
                hopping += 1
                continue

            hopping = 0

            p, path_size, dist_from_p1 = MapPose.project_on_path(poses[i], poses[i + 1], location)

            if p is None or path_size == 0:
                continue

            best_distance = distance_to_segment
            best_segment_size = path_size
            best_dist_from_p1 = dist_from_p1;        

            #only the best cases are to be checked now

            if MapPose.are_close(p, poses[i]):
                best_pos = i
                best_pos_type = NearestPoseType.Exact_P1
                continue
        
            if MapPose.are_close(p, poses[i+1]):
                best_pos = i+1
                best_pos_type = NearestPoseType.Exact_P2
                continue

            # BEFORE THE PATH SEGMENT
            if dist_from_p1 < 0:
                best_pos = i
                best_pos_type = NearestPoseType.Before_P1
                continue

            # AFTER THE PATH SEGMENT
            if dist_from_p1 > path_size:
                best_pos = i+1
                best_pos_type = NearestPoseType.After_P2
                continue            
        
            # INSIDE THE SEGMENT
            if dist_from_p1 <= 0.5 * path_size:
                # p1 is the closest
                best_pos = i
                best_pos_type = NearestPoseType.After_P1
                continue
            else:
                best_pos = i+1
                best_pos_type = NearestPoseType.Before_P2
        
        return NearestPoseSearchResult(best_pos, best_pos_type, best_segment_size, best_dist_from_p1)    

    @classmethod
    def remove_repeated_seq_points_in_list(cls, poses: list['MapPose']) -> list['MapPose']:
        res = []
        res.append(poses[0])
        for i in range(1, len(poses)):
            if poses[i] == poses[i-1]:
                continue
            res.append(poses[i])
        return res

    @classmethod
    def find_nearest_goal_pose(cls, location: 'MapPose', poses: list['MapPose'], start: int = 0, min_distance: float = 10, max_hopping: int = 5) -> int:
        
        nearest_res = MapPose.__find_nearest_pose_in_list(location, poses, start, max_hopping);
 
        last_pos = len(poses) - 1
        dist_from_p2 = 0

        match nearest_res.pos_type:
            case NearestPoseType.NotFound:
                return -2
            case NearestPoseType.Before_P1:
                if (abs(nearest_res.best_distance_from_p1) <= min_distance):
                    if (nearest_res.list_pos == last_pos):
                        return -1
                    else:
                        return nearest_res.list_pos + 1
                else:
                    return nearest_res.list_pos

            case NearestPoseType.Before_P2:
                dist_from_p2 = nearest_res.best_segment_size - nearest_res.best_distance_from_p1

                if (dist_from_p2 <= min_distance):
                    if (nearest_res.list_pos == last_pos):
                        return -1
                    else:
                        return nearest_res.list_pos + 1
                else:
                    return nearest_res.list_pos

            case NearestPoseType.Exact_P1:
                if (nearest_res.list_pos == last_pos):
                    return -1
                return nearest_res.list_pos + 1                
            
            case NearestPoseType.Exact_P2:
                if (nearest_res.list_pos == last_pos):
                    return -1
                return nearest_res.list_pos + 1
            
            case NearestPoseType.After_P1:
                if nearest_res.best_segment_size <= min_distance:
                    # its after P1 but too close to P2
                    if (nearest_res.list_pos + 1 == last_pos):
                        return -1
                    else:
                        return nearest_res.list_pos + 2
                else:
                # We dont need to check if we are at the last pos, because since it is P1, its safe to assume that the next pos is available
                    return nearest_res.list_pos + 1

            case NearestPoseType.After_P2:
                if (nearest_res.list_pos == last_pos):
                    return -1
                else:
                    return nearest_res.list_pos + 1
        return -1
    

##
##  OLD CPP calls: decided to reimplement in Python
##


    # @classmethod
    # def __get_reference(cls, poses: list['MapPose']):
    #     if hasattr(MapPose, "last_poses"):
    #         if MapPose.last_poses is poses:
    #             return MapPose.poses_ref
    #         else:
    #             MapPose.lib.map_pose_free_pose_list(MapPose.poses_ref)
        
    #     MapPose.last_poses = poses
    #     count = len(poses)
    #     data = np.zeros((4 * count), np.float32)
    #     i = 0
    #     for p in poses:
    #         data[i] = p.x
    #         data[i+1] = p.y
    #         data[i+2] = p.z
    #         data[i+3] = p.heading.rad()
    #         i += 4

    #     data = np.ascontiguousarray(data, dtype=np.float32)
        
    #     MapPose.poses_ref = MapPose.lib.map_pose_store_pose_list(data, count)
    #     return MapPose.poses_ref
            
    
    # @classmethod
    # def find_nearest_goal_pose(cls, location: 'MapPose', poses: list['MapPose'], start: int = 0, minDistance: float = 10, maxHopping: int = 5) -> int:
    #     ref = MapPose.__get_reference(poses)
    #     return MapPose.lib.find_best_next_pose_on_list(ref, location.x, location.y, start, minDistance, maxHopping)

 
