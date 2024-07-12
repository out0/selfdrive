from model.world_pose import WorldPose
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from slam.kalman_filter import ExtKalmanFilterLocation

class SLAM:    
    _gps: GPS
    _imu: IMU
    _odometer: Odometer
    _kalman_filter: ExtKalmanFilterLocation

    def __init__(self, gps: GPS, imu: IMU, odometer: Odometer):
        self._gps = gps
        self._imu = imu
        self._odometer = odometer
        self._kalman_filter = ExtKalmanFilterLocation()
        

    def estimate_ego_pose (self) -> WorldPose:
        if self._imu is None or self._odometer is None:
            gps_data = self._gps.read()
            return WorldPose(
                lat=gps_data.latitude,
                lon=gps_data.longitude,
                alt=gps_data.altitude,
                heading=0
            )
        else:
            return self._kalman_filter.estimate_pose(self._gps.read(), self._imu.read(), self._odometer.read())
            