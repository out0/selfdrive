"""
ctypes binding for hermite_interpolation (via the hermite_interpolate_c shim
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

class HermiteInterpolator:
    def __init__(self, lib_path: str | None = None):
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libwpmp.so")

        self._lib = ctypes.CDLL(lib_path)
        self._lib.hermite_interpolate_c.argtypes = [
            ctypes.c_int,                    # plane_width
            ctypes.c_int,                    # plane_height
            ctypes.c_float,                  # p1_x
            ctypes.c_float,                  # p1_z
            ctypes.c_float,                  # p1_heading_rad
            ctypes.c_float,                  # p2_x
            ctypes.c_float,                  # p2_z
            ctypes.c_float,                  # p2_heading_rad
            ctypes.c_float,                  # wheelbase
            ctypes.c_float,                  # delta_max_rad
            ctypes.POINTER(ctypes.c_int),    # out_size
        ]
        self._lib.hermite_interpolate_c.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.hermite_interpolate_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.hermite_interpolate_free.restype = None

    def hermite_interpolation(
        self,
        plane_width: int,
        plane_height: int,
        p1: Waypoint,
        p2: Waypoint,
        wheelbase: float,
        delta_max_rad: float,
    ) -> List[Waypoint]:
        size = ctypes.c_int(0)
        ptr = self._lib.hermite_interpolate_c(
            plane_width, plane_height,
            float(p1.x), float(p1.z), p1.heading.rad(),
            float(p2.x), float(p2.z), p2.heading.rad(),
            wheelbase, delta_max_rad,
            ctypes.byref(size)
        )
        try:
            if not ptr or size.value < 3:
                # size == 1 is the C++ side's "failed / fake" sentinel value
                return []
            flat = ptr[:size.value]
            return [
                Waypoint(int(round(flat[i])), int(round(flat[i + 1])), angle.new_rad(flat[i + 2]))
                for i in range(0, len(flat), 3)
            ]
        finally:
            if ptr:
                self._lib.hermite_interpolate_free(ptr)