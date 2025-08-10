import numpy as np
from pydriveless import SearchFrame, MapPose, Waypoint
import json

class PlanningData:
    __seq: int
    __og: SearchFrame
    __base_map_conversion_location: MapPose
    __ego_location: MapPose
    __start: Waypoint
    __g1: MapPose
    __g2: MapPose
    __velocity: float
    __min_distance_px: tuple[int, int]
    
    def __init__(self, 
                 seq: int, 
                 og: SearchFrame, 
                 ego_location: MapPose, 
                 g1: MapPose, 
                 g2: MapPose,
                 start: Waypoint,
                 velocity: float,
                 min_distance: tuple[int, int],
                 base_map_conversion_location: MapPose = None):
        self.__seq = seq
        self.__og = og
        self.__ego_location = ego_location
        self.__g1 = g1
        self.__g2 = g2
        self.__start = start
        self.__velocity = velocity
        self.__min_distance_px = min_distance
        self.__local_goal = None
        self.__base_map_conversion_location = ego_location if base_map_conversion_location is None else base_map_conversion_location
        
    def og(self) -> SearchFrame:
        return self.__og

    def ego_location(self) -> MapPose:
        return self.__ego_location

    def g1(self) -> MapPose:
        return self.__g1

    def g2(self) -> MapPose:
        return self.__g2
    
    def start(self) -> Waypoint:
        return self.__start

    def velocity(self) -> float:
        return self.__velocity
    
    def min_distance(self) -> tuple[int, int]:
        return self.__min_distance_px
    
    def set_local_goal(self, l1: Waypoint) -> None:
        self.__local_goal = l1
    
    def local_goal(self) -> Waypoint:
        return self.__local_goal
    
    @property
    def base_map_conversion_location(self) -> MapPose:
        return self.__base_map_conversion_location
        
    def __str__(self):
        lb = self.__og.lower_bound()
        ub = self.__og.upper_bound()
        return json.dumps({
            "seq": self.__seq,
            "velocity": self.__velocity,
            "min_distance_px": self.__min_distance_px,
            "ego_location": str(self.__ego_location),
            "g1": str(self.__g1),
            "g2": str(self.__g2),
            "local_goal": str(self.__local_goal) if self.__local_goal else None,
            "og_width": self.__og.width(),
            "og_height": self.__og.height(),
            "og_lower_bound_x": str(lb[0]),
            "og_lower_bound_z": str(lb[1]),
            "og_upper_bound_x": str(ub[0]),
            "og_upper_bound_z": str(ub[1])
        })
    
    def from_str(val: str) -> 'PlanningData':
        data = json.loads(val)
        lb = (int(data['og_lower_bound_x']), int(data['og_lower_bound_z']))
        ub = (int(data['og_upper_bound_x']), int(data['og_upper_bound_z']))
        res = PlanningData(
            seq=int(data['seq']),
            velocity=float(data['velocity']),
            min_distance=tuple(data['min_distance_px']),
            ego_location=MapPose.from_str(data['ego_location']),
            g1=MapPose.from_str(data['g1']),
            g2=MapPose.from_str(data['g2']),
            og=SearchFrame(
                width=data['og_width'],
                height=data['og_height'],
                lower_bound=lb,
                upper_bound=ub
            )
        )
        res.set_local_goal(Waypoint.from_str(data['local_goal']) if data['local_goal'] else None)
        return res

    @property
    def seq(self) -> int:
        return self.__seq