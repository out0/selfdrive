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
from pydriveless import Waypoint
import os


# bool (*)(void*, int, int, float) -- matches interpolation_callback in the C++ code
_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_bool,  # return
    ctypes.c_void_p,  # ctx
    ctypes.c_int,  # x
    ctypes.c_int,  # z
    ctypes.c_float,  # heading
)


class HermiteInterpolator:
    """Thin wrapper around hermite_interpolate_c exposed by libhermite.so."""

    def __init__(self, lib_path: str | None = None):
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libwpmp.so")
                   
        self._lib = ctypes.CDLL(lib_path)
        self._lib.hermite_interpolate_c.argtypes = [
            ctypes.c_int,  # plane_width
            ctypes.c_int,  # plane_height
            ctypes.c_float,  # p1_x
            ctypes.c_float,  # p1_z
            ctypes.c_float,  # p1_heading_rad
            ctypes.c_float,  # p2_x
            ctypes.c_float,  # p2_z
            ctypes.c_float,  # p2_heading_rad
            ctypes.c_float,  # wheelbase
            ctypes.c_float,  # delta_max_rad
            _CALLBACK_TYPE,  # cb
            ctypes.c_void_p,  # result_ptr (unused on the Python side, kept for API parity)
        ]
        self._lib.hermite_interpolate_c.restype = ctypes.c_bool

    def hermite_interpolation(
        self,
        plane_width: int,
        plane_height: int,
        p1: Waypoint,
        p2: Waypoint,
        wheelbase: float,
        delta_max_rad: float,
    ) -> List[Waypoint]:
        """
        Interpolates a hermite curve between p1 and p2.

        Returns a list of Waypoints, or an empty list if the curve exceeded
        the steering limit (mirroring `res.clear()` in the original C++ code).
        """
        collected: List[Waypoint] = []

        # The callback runs synchronously inside the C++ call below (same
        # thread, same call stack), so it's safe to simply append to the
        # Python list captured by closure. No ctx/void* juggling needed.
        def _collect(ctx, x, z, heading) -> bool:
            collected.append(Waypoint(x, z, heading))
            return True  # True = keep interpolating

        c_callback = _CALLBACK_TYPE(_collect)

        valid = self._lib.hermite_interpolate_c(
            plane_width,
            plane_height,
            float(p1.x),
            float(p1.z),
            float(p1.heading.rad()),
            float(p2.x),
            float(p2.z),
            float(p2.heading.rad()),
            float(wheelbase),
            float(delta_max_rad),
            c_callback,
            None,  # ctx not needed: Python closure replaces the void* trick
        )

        if not valid:
            collected.clear()

        return collected
