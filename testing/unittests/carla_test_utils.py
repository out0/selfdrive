import sys
sys.path.append("../../../")
from model.planning_data import PlanningData
import cv2, numpy as np, json
from vision.occupancy_grid_cuda import SEGMENTED_COLORS
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from carlasim.carla_client import CarlaClient
import carla

class CarlaTestUtils:
    _client: CarlaClient
    
    def __init__(self, client: CarlaClient):
        self._client = client
        

    def show_path(self, path: list[MapPose]):
        world = self._client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=12000.0,
                                        persistent_lines=True)

    def show_point(self, x: float, y: float) -> None:
        world = self._client.get_world()
        world.debug.draw_string(carla.Location(x, y, 2), 'X', draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=12000.0,
                                        persistent_lines=True)