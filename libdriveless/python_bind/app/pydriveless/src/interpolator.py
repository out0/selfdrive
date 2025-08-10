import math
import ctypes
import numpy as np
from .angle import angle
import os
from .waypoint import Waypoint
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
    def hermite(cls, width: int, height: int, p1: Waypoint, p2: Waypoint, return_as_waypoint: bool = True) -> Union[list[Waypoint], np.ndarray]:
        Interpolator.setup_cpp_lib()
        raw_res = Interpolator.lib.interpolate_hermite(width, height, p1.x,
                                                       p1.z, p1.heading.rad(), p2.x, p2.z, p2.heading.rad())
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
