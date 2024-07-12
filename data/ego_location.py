import sys
sys.path.append("..")

from model.world_pose import WorldPose
from model.sensors.gps import GPS
from model.sensors.odometer import Odometer
from model.sensors.imu import IMU
from model.ego_car import EgoCar
from model.sensor_data import *

class EgoLocation:
        _gps: GPS
        _odometer: Odometer
        _imu: IMU
    
        def __init__(self, ego: EgoCar):
            self._gps = ego.get_gps()
            self._odometer = ego.get_odometer()
            self._imu = ego.get_imu()
        
        def estimate_ego_pose(self) -> WorldPose:
            gps_data = self._gps.read()
            imu_data = self._imu.read()

            #
            # TODO: add Kalman Filter to handle sensor fault
            #
            return WorldPose(
                gps_data.latitude,
                gps_data.longitude,
                gps_data.altitude,
                imu_data.heading
            )
        
        def ego_velocity(self) -> float:
            return self._odometer.read()
        
        
    