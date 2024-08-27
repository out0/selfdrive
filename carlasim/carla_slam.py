from slam.slam import SLAM
from model.map_pose import MapPose
from carlasim.carla_ego_car import EgoCar
from model.world_pose import WorldPose
from data.coordinate_converter import CoordinateConverter
import math

class CarlaSLAM (SLAM):
    _car: EgoCar
    _coordinate_converter: CoordinateConverter

    def __init__(self, car: EgoCar) -> None:
        self._car = car
        pass

    def estimate_ego_pose(self) -> MapPose:
        location = self._car.get_location()
        heading = self._car.get_heading()
        pose = MapPose(x=location[0], y=location[1], z=0, heading=heading)
        pose.z = location[2]
        return pose

    def calibrate(self) -> None:
        pose = self.__read_raw_world_data()
        self._coordinate_converter = CoordinateConverter(pose)

    def manual_calibrate(self, pose: WorldPose) -> None:
        self._coordinate_converter = CoordinateConverter(pose)

    def get_coordinate_converter(self) -> CoordinateConverter:
        return self._coordinate_converter
  
    def read_gps(self) -> WorldPose:
        gps_data = self._car.get_gps().read()
        if gps_data is None:
            return None
        
        return WorldPose(
            lat=gps_data.latitude,
            lon=gps_data.longitude,
            alt=gps_data.altitude,
            heading=0
        )
