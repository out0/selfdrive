import math
import ctypes
import numpy as np
from .angle import angle
import os
from .waypoint import Waypoint
from .map_pose import MapPose
from .search_params import SearchParams, EgoParams
from typing import Union


class Interpolator:

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(Interpolator, "lib"):
            return

        lib_path = os.path.join(os.path.dirname(
            __file__), "../cpp", "libdriveless.so")
        Interpolator.lib = ctypes.CDLL(lib_path)

        Interpolator.lib.interpolate_hermite.restype = ctypes.POINTER(
            ctypes.c_float)        
        Interpolator.lib.interpolate_hermite.argtypes = [
            ctypes.c_int,    # width
            ctypes.c_int,    # height
            ctypes.c_int,    # x1
            ctypes.c_int,    # z1
            ctypes.c_float,  # h1_rad
            ctypes.c_int,    # x2
            ctypes.c_int,    # z2
            ctypes.c_float,  # h2_rad
            ctypes.c_float   # max_curvature
        ]

        Interpolator.lib.interpolate_cubic_spline.restype = ctypes.POINTER(
            ctypes.c_float)
        Interpolator.lib.interpolate_cubic_spline.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1),   # data points
            ctypes.c_int                                            # resolution
        ]

        Interpolator.lib.free_interpolation_arr.restype = None
        Interpolator.lib.free_interpolation_arr.argtypes = [
            ctypes.POINTER(ctypes.c_float)
            
        ]

    def __convert_raw_arr_to_waypoint_list(raw_res: any) -> list[Waypoint]:
        count = int(raw_res[0])
        res = []
        for i in range(count):
            pos = 3 * i + 1
            res.append(Waypoint(
                x=float(raw_res[pos]),
                z=float(raw_res[pos+1]),
                heading=angle.new_rad(float(raw_res[pos+2]))
            ))
        return res

    def __convert_float_arr_to_numpy(arr: any) -> np.ndarray:
        count = int(arr[0])
        res = np.zeros((count, 3), dtype=np.float32)
        for i in range(count):
            pos = 3*i + 1
            res[i, 0] = float(arr[pos])
            res[i, 1] = float(arr[pos+1])
            res[i, 2] = float(arr[pos+2])
        return res

    def __convert_list_waypoint_to_np_array(arr: list[Waypoint]) -> np.ndarray:
        count = len(arr)
        res = np.zeros((3 * count + 1), dtype=np.float32)
        res[0] = count
        for i in range(count):
            pos = 3 * i + 1
            res[pos] = arr[i].x
            res[pos + 1] = arr[i].z
            res[pos + 2] = arr[i].heading.rad()
        return res

    @classmethod
    def hermite(cls, width: int, height: int, p1: Waypoint, p2: Waypoint, return_as_waypoint: bool = True, max_curvature: float = -1) -> Union[list[Waypoint], np.ndarray]:
        Interpolator.setup_cpp_lib()
        raw_res = Interpolator.lib.interpolate_hermite(width, height, p1.x,
                                                       p1.z, p1.heading.rad(), p2.x, p2.z, p2.heading.rad(), max_curvature)
        if return_as_waypoint:
            res = Interpolator.__convert_raw_arr_to_waypoint_list(raw_res)
        else:
            res = Interpolator.__convert_float_arr_to_numpy(raw_res)

        Interpolator.lib.free_interpolation_arr(raw_res)
        return res

    @classmethod
    def cubic_spline(cls, control_points: list[Waypoint], resolution: int = 10, return_as_waypoint: bool = True) -> Union[list[Waypoint], np.ndarray]:
        Interpolator.setup_cpp_lib()
        np_arr = Interpolator.__convert_list_waypoint_to_np_array(
            control_points)
        f = np.ascontiguousarray(np_arr, dtype=np.float32)
        raw_res = Interpolator.lib.interpolate_cubic_spline(f, resolution)

        if return_as_waypoint:
            res = Interpolator.__convert_raw_arr_to_waypoint_list(raw_res)
        else:
            res = Interpolator.__convert_float_arr_to_numpy(raw_res)

        Interpolator.lib.free_interpolation_arr(raw_res)
        return res

    @classmethod
    def bicycle_model(cls, ego_params: EgoParams, search_params: SearchParams, steering_angle: angle, path_size_px: int = -1, reverse: bool = False) -> tuple[list[MapPose], list[Waypoint]]:
        """ Generate path from the center of gravity
        """
        v = search_params.velocity_m_s
        a = steering_angle.rad()
        
        if reverse:
            a += math.pi

        steer = math.tan(a)
        
        ego_location = search_params.ego_pose
        x = ego_location.x
        y = ego_location.y
        heading = ego_location.heading.rad()
        path = []
        local_path = []

        lr = 0.5 * ego_params.vehicle_length_m
        dt = 0.05

        if path_size_px > 0:
            steps = path_size_px / dt
        else:
            steps = search_params.max_path_size_px / dt

        conv = ego_params.coordinate_converter()
        base_location = search_params.map_origin
        w, h = ego_params.search_frame_dimensions

        for _ in range (0, steps):
            beta = math.atan(steer / lr)
            x += v * math.cos(heading + beta) * dt
            y += v * math.sin(heading + beta) * dt
            heading += v * math.cos(beta) * steer * dt / (ego_params.vehicle_length_m)
            next_point = MapPose(x, y, ego_location.z, heading=heading, reversed=reverse)
            next_point_local = conv.convert(base_location, next_point)
            
            if next_point_local.x >= w or next_point_local.x < 0:
                return (path, local_path)

            if next_point_local.z >= h or next_point_local.z < 0:
                return (path, local_path)

            path.append(next_point)
            local_path.append(next_point_local)

        return (path, local_path)