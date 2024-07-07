import sys
sys.path.append("..")

from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters
import math, numpy as np

EARTH_RADIUS = 6378137.0

class CoordinateConverter:
    
    _world_origin: WorldPose
    _orig_heading: float

    def __init__(self, world_origin: WorldPose):
        self._world_origin = world_origin
        
    def __convert_map_heading_to_compass(h: float) -> float:
        return (h + 90 + 360) % 360

    def __convert_compass_to_map_heading(hc: float) -> float:
        p = (hc - 90 - 360)  % 360
        if p > 180:
            return p - 360
        return p
    

    def convert_to_map_pose(self, world_pose: WorldPose) -> MapPose:
        scale = math.cos(math.radians(self._world_origin.lat))
        return MapPose(
            x=scale * EARTH_RADIUS * math.radians(world_pose.lon),
            y=-scale * EARTH_RADIUS * math.log(math.tan(math.pi * (90 + world_pose.lat) / 360)),
            z=world_pose.alt,
            heading=CoordinateConverter.__convert_compass_to_map_heading(world_pose.heading)
        )
        
    def convert_to_world_pose(self, map_pose: MapPose) -> WorldPose:
        scale = math.cos(math.radians(self._world_origin.lat))  
        return WorldPose(
            lat=360 * math.atan(math.exp(-map_pose.y / (EARTH_RADIUS * scale))) / math.pi - 90,
            lon=map_pose.x * 180 / (math.pi * EARTH_RADIUS * scale),
            alt=map_pose.z,
            heading=CoordinateConverter.__convert_map_heading_to_compass(map_pose.heading)
        )
        
    # def convert_to_map_pose(self, world_pose: WorldPose) -> MapPose:        
    #     [x, y] = self.__carla_convert_to_map_pose(world_pose.lat, world_pose.lon)
    #     return MapPose(
    #         x=x, 
    #         y=y, 
    #         z=world_pose.alt, 
    #         heading=CoordinateConverter.__convert_compass_to_map_heading(world_pose.heading)
    #     )
        
    # def convert_to_world_pose(self, map_pose: MapPose) -> WorldPose:        
    #     [lat, lon] = self.__carla_convert_to_world_pose(map_pose.x, map_pose.y)
    #     return MapPose(
    #         lat=lat, 
    #         lon=lon, 
    #         alt=map_pose.z, 
    #         heading=CoordinateConverter.__convert_map_heading_to_compass(map_pose.heading)
    #     )
        
    
    # # Runs ok in Carla but I'm not sure if it will run in real hardware!
    # # copied code; seems not based on mathematics....
    # # https://github.com/carla-simulator/carla/issues/3871
    
    # def __carla_convert_to_map_pose(self, lat, lon) -> tuple[float, float]:
    #     lat_rad = (math.radians(lat) + np.pi) % (2 * np.pi) - np.pi
    #     lon_rad = (math.radians(lon) + np.pi) % (2 * np.pi) - np.pi
    #     #lat_rad = lat
    #     #lon_rad = lon
    #     R = 6378135 # Aequatorradii
    #     x = R * np.sin(lon_rad) * np.cos(lat_rad)       # iO
    #     y = R * np.sin(-lat_rad)                               # iO
    #     return [x, y]
    
    # def __carla_convert_to_world_pose(self, x: float, y: float) -> tuple[float, float]:
    #     R = 6378135 # Aequatorradii
    #     lat = -math.asin(y/R)
    #     if lat == 0:
    #         lat = 1e-11
    #     lon = math.acos(x / (R * math.sin(lat)))
    #     return math.degrees(lat), math.degrees(lon)
