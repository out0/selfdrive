import carla
from carlasim.carla_client import CarlaClient
from model.map_pose import MapPose

class CarlaDebug:
    _client: CarlaClient
    
    def __init__(self, client: CarlaClient) -> None:
        self._client = client
        pass
    
    def show_map_pose(self, p: MapPose):
        world = self._client.get_world()
        world.debug.draw_string(carla.Location(p.x, p.y, 2), 'X', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=1200.0,
                                persistent_lines=True)

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