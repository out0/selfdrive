from slam.slam import SLAM
from model.map_pose import MapPose
from carlasim.carla_ego_car import EgoCar
import math

class CarlaSLAM (SLAM):
    _car: EgoCar

    def __init__(self, car: EgoCar) -> None:
        self._car = car
        pass

    def estimate_ego_pose(self) -> MapPose:
        location = self._car.get_location()
        heading = self._car.get_heading()
        pose = MapPose(x=location[0], y=location[1], z=0, heading=heading)
        pose.z = location[2]
        return pose

