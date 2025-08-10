import math, ctypes
from .angle import angle
import os

class WorldPose:    
    __lat: angle  # degrees
    __lon: angle  # degrees
    __alt: float  # meters
    __compass: angle  # degrees
    
    def __init__(self, 
                 lat: angle, 
                 lon: angle, 
                 alt: float, 
                 compass: angle):
        self.__lat = lat
        self.__lon = lon
        self.__alt = alt
        self.__compass = compass
        WorldPose.setup_cpp_lib()
    
    @property
    def lat(self) -> angle:
        return self.__lat
    
    @property
    def lon(self) -> angle:
        return self.__lon

    @property
    def alt(self) -> float:
        return self.__alt
 
    @property
    def compass(self) -> angle:
        return self.__compass
    
    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(WorldPose, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
            
        WorldPose.lib = ctypes.CDLL(lib_path)

        WorldPose.lib.world_pose_distance_between.restype = ctypes.c_double
        WorldPose.lib.world_pose_distance_between.argtypes = [
            ctypes.c_double, # lat1
            ctypes.c_double, # lon1
            ctypes.c_double, # lat2
            ctypes.c_double, # lon2
        ]
        WorldPose.lib.world_pose_compute_heading.restype = ctypes.c_double
        WorldPose.lib.world_pose_compute_heading.argtypes = [
            ctypes.c_double, # lat1
            ctypes.c_double, # lon1
            ctypes.c_double, # lat2
            ctypes.c_double, # lon2
        ]
        
    
    # O(logn)
    def distance_between(p1 : 'WorldPose', p2 : 'WorldPose') -> float:
        """Compute the Haversine distance between two world absolute poses (ignoring heading)

        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            float: distance in meters
        """
        return WorldPose.lib.world_pose_distance_between(
            p1.lat.rad(),
            p1.lon.rad(),
            p2.lat.rad(),
            p2.lon.rad()
        )
                

    def compute_heading(p1 : 'WorldPose', p2 : 'WorldPose') -> angle:
        """Computes the bearing (World heading or forward Azimuth) for two world poses

        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            float: angle in radians
        """
        return angle.new_rad(WorldPose.lib.world_pose_compute_heading(
            p1.lat.rad(),
            p1.lon.rad(),
            p2.lat.rad(),
            p2.lon.rad()
        ))

    def from_str(payload: str) -> 'WorldPose':
        p = payload.split("|")
        return WorldPose(
            angle.new_deg(float(p[0])),
            angle.new_deg(float(p[1])),
            float(p[2]),
            angle.new_deg(float(p[3]))
        )

    def __str__(self) -> str:
        return f"{self.lat.deg()}|{self.lon.deg()}|{self.alt}|{self.compass.deg()}"
