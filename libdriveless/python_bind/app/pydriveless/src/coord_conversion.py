import math
from .angle import angle, QUARTER_PI, HALF_PI, PI, DOUBLE_PI
from .map_pose import MapPose
from .world_pose import WorldPose
from .waypoint import Waypoint
from typing import Union

EARTH_RADIUS = 6378137.0

class CoordinateConverter:
    
    __world_coord_scale: float
    __origin_compass_angle: angle
    __rateW: float
    __rateH: float
    __invRateW: float
    __invRateH: float
    __center: Waypoint
    
    def __init__(self, 
                 origin: WorldPose, 
                 width: int, 
                 height: int, 
                 perceptionWidthSize_m: float, 
                 perceptionHeightSize_m: float):
        self.__world_coord_scale = math.cos(origin.lat.rad())
        self.__origin_compass_angle = origin.compass
        self.__rateW = width / perceptionWidthSize_m
        self.__rateH = height / perceptionHeightSize_m
        self.__invRateW = perceptionWidthSize_m / width
        self.__invRateH = perceptionHeightSize_m / height
        self.__center = Waypoint(int(width/2), int(height/2))
    
    
    def __convert_map_heading_to_compass(self, h: float) -> float:
        a = h + self.__origin_compass_angle.rad()
        if a > PI: a -= DOUBLE_PI
        
        if a < 0: return a + DOUBLE_PI
        return a


    def __convert_compass_to_map_heading(self, hc: float) -> float:
        a = hc - self.__origin_compass_angle.rad()
        if a < 0:  # compass should never be < 0
            a += DOUBLE_PI
        if a > PI: return a - DOUBLE_PI
        return a 



    def __convertWorldToMap(self, pose: WorldPose) -> MapPose:
        x = self.__world_coord_scale * EARTH_RADIUS * pose.lon.rad()
        y = -self.__world_coord_scale * EARTH_RADIUS * math.log(math.tan(QUARTER_PI + 0.5 * pose.lat.rad()))
        h = self.__convert_compass_to_map_heading(pose.compass.rad())
        return MapPose(x, y, pose.alt, angle.new_rad(h))

    def __convertMapToWorld(self, pose: MapPose) -> WorldPose:
        lat = 2 * math.atan(math.exp(-pose.y/(self.__world_coord_scale * EARTH_RADIUS))) - HALF_PI
        lon = pose.x / (self.__world_coord_scale * EARTH_RADIUS)
        return WorldPose(
            angle.new_rad(lat),
            angle.new_rad(lon),
            pose.z,
            angle.new_rad(self.__convert_map_heading_to_compass(pose.heading.rad())))

    def convertWorld(self, pose: Union[WorldPose, MapPose]) -> Union[MapPose, WorldPose]:
        if isinstance(pose, WorldPose):
            return self.__convertWorldToMap(pose)
        else:
            return self.__convertMapToWorld(pose)
    
    
    def __convertMapToWaypoint(self, location: MapPose, target: MapPose) -> Waypoint:
        dx = target.x - location.x
        dy = target.y - location.y
        c = math.cos(-location.heading.rad())
        s = math.sin(-location.heading.rad())
        p0 = self.__rateH * (c * dx - s * dy)
        p1 = self.__rateW * (s * dx + c * dy)
        x = int(round(self.__center.x + p1))
        z = int(round(self.__center.z - p0))

        return Waypoint(x, z, target.heading - location.heading)
    
    def __convertWaypointToMap(self, location: MapPose, target: Waypoint) -> MapPose:
        p0 = self.__center.z - target.z
        p1 = target.x - self.__center.x
        c = math.cos(location.heading.rad())
        s = math.sin(location.heading.rad())

        x = self.__invRateH * c * p0 - self.__invRateW * s * p1 + location.x
        y = self.__invRateH * s * p0 + self.__invRateW * c * p1 + location.y

        return MapPose(
            x,
            y,
            location.z,
            target.heading + location.heading)
    
    def convert(self, location: MapPose, pose: Union[MapPose, Waypoint]) -> Union[MapPose, Waypoint]:
        if isinstance(pose, MapPose):
            return self.__convertMapToWaypoint(location, pose)
        else:
            return self.__convertWaypointToMap(location, pose)
    
    def convert_list_map_to_waypoint(self, location: MapPose, list_map: list[MapPose]) -> list[Waypoint]:
        res = []
        for p in list_map:
            res.append(self.__convertMapToWaypoint(location, p))
        return res
    
    def convert_list_waypoint_to_map(self, location: MapPose, list_waypoint: list[Waypoint]) -> list[MapPose]:
        res = []
        for p in list_waypoint:
            res.append(self.__convertWaypointToMap(location, p))
        return res    