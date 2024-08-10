import sys
sys.path.append("..")

from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.physical_parameters import PhysicalParameters
import math, numpy as np

EARTH_RADIUS = 6378137.0

class CoordinateConverter:
    
    _lat_origin: float
    _map_pose_origin: MapPose
    _orig_heading: float
    _rw: float
    _rh: float
    _og_center: Waypoint

    def __init__(self, world_origin: WorldPose):
        self._lat_origin = world_origin.lat
        self._map_pose_origin = self.convert_world_to_map_pose(world_origin)
        self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
        self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        self._og_center = Waypoint(math.floor(PhysicalParameters.OG_WIDTH/2), math.floor(PhysicalParameters.OG_HEIGHT/2))

        
    def __convert_map_heading_to_compass(h: float) -> float:
        return (h + 90 + 360) % 360

    def __convert_compass_to_map_heading(hc: float) -> float:
        p = (hc - 90 - 360)  % 360
        if p > 180:
            return p - 360
        return p
    
    def get_relative_map_pose(self, world_pose: WorldPose) -> MapPose:
        pose = self.convert_world_to_map_pose(world_pose)
        return pose - self._map_pose_origin
    
    def get_relative_world_pose(self, map_pose: MapPose) -> WorldPose:
        m = map_pose + self._map_pose_origin
        pose = self.convert_map_to_world_pose(m)
        return pose
    
    def convert_world_to_map_pose(self, world_pose: WorldPose) -> MapPose:
        scale = math.cos(math.radians(self._lat_origin))

        return MapPose(
            x=scale * EARTH_RADIUS * math.radians(world_pose.lon),
            y=-scale * EARTH_RADIUS * math.log(math.tan(math.pi * (90 + world_pose.lat) / 360)),
            z=world_pose.alt,
            heading=CoordinateConverter.__convert_compass_to_map_heading(world_pose.heading)
        )
        
    def convert_map_to_world_pose(self, map_pose: MapPose) -> WorldPose:
        scale = math.cos(math.radians(self._lat_origin))  
        return WorldPose(
            lat=360 * math.atan(math.exp(-map_pose.y / (EARTH_RADIUS * scale))) / math.pi - 90,
            lon=map_pose.x * 180 / (math.pi * EARTH_RADIUS * scale),
            alt=map_pose.z,
            heading=CoordinateConverter.__convert_map_heading_to_compass(map_pose.heading)
        )
    
    def __build_translation_mat(self, x: float, y: float) -> np.ndarray:
        return np.array([
            [1, 0 , 0],
            [0, 1, 0],
            [x, y, 1]
        ])

    def __build_rotation_mat(self, angle: float) -> np.ndarray:
        r = math.radians(angle)
        c = math.cos(r)
        s = math.sin(r)
    
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0 , 0, 1]
        ])

    def __build_resize_mat(self, ratio_x: float, ratio_y) -> np.ndarray:
        return np.array([
            [ratio_x, 0, 0],
            [0, ratio_y, 0],
            [0 , 0, 1]
        ])

    def convert_map_to_waypoint(self, location: MapPose, target: MapPose) -> Waypoint:
        m = self.__build_translation_mat(-location.x, -location.y) @\
            self.__build_rotation_mat(-location.heading) @\
            self.__build_resize_mat(self._rh, self._rw)
        
        p = np.array([target.x, target.y, 1]) @ m

        x = self._og_center.x + p[1]
        z = self._og_center.z - p[0]
        return Waypoint(x, z)

    def  convert_map_path_to_waypoint(self, location: MapPose, target_list: list[MapPose]) -> list[Waypoint]:
        m = self.__build_translation_mat(-location.x, -location.y) @\
            self.__build_rotation_mat(-location.heading) @\
            self.__build_resize_mat(self._rh, self._rw)
        
        res = []
        for target in target_list:
            p = np.array([target.x, target.y, 1]) @ m

            x = self._og_center.x + p[1]
            z = self._og_center.z - p[0]
    
            res.append(Waypoint(math.floor(x), math.floor(z)))
        return res

    def convert_waypoint_path_to_map_pose(self, location: MapPose, path : list[Waypoint]) -> list[MapPose]:

        m = self.__build_resize_mat(1/self._rh, 1/self._rw) @\
            self.__build_rotation_mat(location.heading) @\
            self.__build_translation_mat(location.x, location.y)
        
        res = []
        for waypoint in path:
            p = np.array([self._og_center.z - waypoint.z, waypoint.x - self._og_center.x, 1]) @ m
            res.append(MapPose(p[0], p[1], 0, 0))
        
        return res
        

    def convert_waypoint_to_map_pose(self, location: MapPose, target : Waypoint) -> MapPose:

        p = np.array([self._og_center.z - target.z,
                      target.x - self._og_center.x, 
                      1])
    
        m = self.__build_resize_mat(1/self._rh, 1/self._rw) @\
            self.__build_rotation_mat(location.heading) @\
            self.__build_translation_mat(location.x, location.y)
        
        p = p @ m

        return MapPose(p[0], p[1], 0, 0)

    def clip (self, p: Waypoint) -> Waypoint:
        if p.x < 0:
            p.x = 0
        elif p.x >= PhysicalParameters.OG_WIDTH:
            p.x = PhysicalParameters.OG_WIDTH - 1
        if p.z < 0:
            p.z = 0
        elif p.z >= PhysicalParameters.OG_HEIGHT:
            p.z = PhysicalParameters.OG_HEIGHT - 1
        return p
    
    def waypoint_convert_ratio (self, w: float, h: float) -> tuple[float, float]:
        return w * self._rw, h * self._rh
    
    
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
