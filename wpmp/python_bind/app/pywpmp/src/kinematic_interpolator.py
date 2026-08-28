"""
ctypes binding for kinematic_interpolation (via the hermite_interpolate_c shim
in hermite_c_api.cpp).

Build the shared library first, e.g.:
    g++ -shared -fPIC -O2 -o libhermite.so hermite_c_api.cpp \
        -I/path/to/driveless/include -L/path/to/driveless/lib -ldriveless

Then place libhermite.so next to this file (or pass an explicit path to
HermiteInterpolator()).
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List
from pydriveless import Waypoint, angle
import os


# bool (*)(void*, int, int, float) -- matches interpolation_callback in the C++ code
_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_bool,  # return
    ctypes.c_void_p,  # ctx
    ctypes.c_int,  # x
    ctypes.c_int,  # z
    ctypes.c_float,  # heading
)


class KinematicInterpolator:
    """Thin wrapper around hermite_interpolate_c exposed by libhermite.so."""

    def __init__(self, lib_path: str | None = None):
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libwpmp.so")
                   
        self._lib = ctypes.CDLL(lib_path)
        self._lib.kinematic_interpolate_c.argtypes = [
            ctypes.c_int,                    # plane_width
            ctypes.c_int,                    # plane_height
            ctypes.c_int,                    # x
            ctypes.c_int,                    # z
            ctypes.c_float,                  # heading
            ctypes.c_float,                  # steering_angle
            ctypes.c_int,                    # max_path_size_px
            ctypes.c_int,                    # wheelbase_px
            ctypes.POINTER(ctypes.c_int),    # out_size
        ]
        self._lib.kinematic_interpolate_c.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.kinematic_interpolate_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.kinematic_interpolate_free.restype = None

    def kinematic_interpolation(
        self,
        plane_width: int,
        plane_height: int,
        p1: Waypoint,
        steering_angle: angle,
        max_path_size: float,
        wheelbase_px: float
    ) -> List[Waypoint]:
        """
        Interpolates a Kinematic curve from p1.

        Returns a list of Waypoints, or an empty list if the curve exceeded
        the steering limit (mirroring `res.clear()` in the original C++ code).
        """
        size = ctypes.c_int(0)
        ptr = self._lib.kinematic_interpolate_c(
            plane_width, plane_height, p1.x, p1.z,
            p1.heading.rad(), steering_angle.rad(), max_path_size, wheelbase_px,
            ctypes.byref(size)
        )
        try:
            if size.value == 0:
                return []

            flat = ptr[:size.value]  # copies data into a Python list of floats
            # cb() pushes (x, z, heading) triplets per point
            waypoints = []
            for i in range(0, len(flat), 3):
                x, z, heading = flat[i], flat[i + 1], flat[i + 2]
                waypoints.append(Waypoint(int(round(x)), int(round(z)), angle.new_rad(heading)))
            return waypoints
        finally:
            self._lib.kinematic_interpolate_free(ptr)