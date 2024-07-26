from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from data.coordinate_converter import CoordinateConverter

class SLAM:    
    _gps: GPS
    _imu: IMU
    _odometer: Odometer
    _coordinate_converter: CoordinateConverter

    def __init__(self, gps: GPS, imu: IMU, odometer: Odometer):
        self._gps = gps
        self._imu = imu
        self._odometer = odometer
        self._coordinate_converter = None
        
    def initialize(self) -> None:
        pose = self.read_gps()
        self._coordinate_converter = CoordinateConverter(pose)

    def read_gps(self) -> WorldPose:
        gps_data = self._gps.read()
        return WorldPose(
            lat=gps_data.latitude,
            lon=gps_data.longitude,
            alt=gps_data.altitude,
            heading=0
        )


    def estimate_ego_pose (self) -> MapPose:
        pose: WorldPose = None
        
        if self._imu is None or self._odometer is None:
            pose = self.read_gps()
        else:
            # TODO
            #return self._kalman_filter.estimate_pose(self._gps.read(), self._imu.read(), self._odometer.read())
            return None
        
        return self._coordinate_converter.convert_to_map_pose(pose)