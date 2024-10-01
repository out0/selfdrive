import carla
from carlasim.carla_client import CarlaClient
from model.map_pose import MapPose
import math


class CarlaDebug:
    _client: CarlaClient
    
    def __init__(self, client: CarlaClient) -> None:
        self._client = client
        pass
    
    def show_map_pose(self, p: MapPose, show_heading: bool = False, lifetime = -1, mark: str = 'X'):
        world = self._client.get_world()
        world.debug.draw_string(carla.Location(p.x, p.y, 4), mark, draw_shadow=True,
                                color=carla.Color(r=0, g=0, b=255), life_time=lifetime,
                                persistent_lines=True)
               
        if show_heading:
            d = 5
            h = math.radians(p.heading)
            end_location = carla.Location(p.x + d * math.cos(h), p.y + d * math.sin(h), 2)
            world.debug.draw_arrow(begin=carla.Location(p.x, p.y, 2), end=end_location, thickness=0.1, color=carla.Color(0,255,0), life_time=lifetime)

    def show_path(self, path: list[MapPose]):
        world = self._client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1200.0,
                                        persistent_lines=True)
                
    def show_global_path(self, path: list[MapPose]):
        world = self._client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'X', draw_shadow=False,
                                    color=carla.Color(r=0, g=0, b=255), life_time=1200.0,
                                    persistent_lines=True)
            
    