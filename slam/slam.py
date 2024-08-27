from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
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
    
    def __read_raw_world_data(self) -> WorldPose:
        w = self.read_gps()
        if w is None:
            return None
        
        imu_data = self._imu.read()
        w.heading = imu_data.compass
        return w
    
    def calibrate(self) -> None:
        pose = self.__read_raw_world_data()
        self._coordinate_converter = CoordinateConverter(pose)

    def manual_calibrate(self, pose: WorldPose) -> None:
        self._coordinate_converter = CoordinateConverter(pose)

    def get_coordinate_converter(self) -> CoordinateConverter:
        return self._coordinate_converter

    def read_gps(self) -> WorldPose:
        gps_data = self._gps.read()
        if gps_data is None:
            return None
        
        return WorldPose(
            lat=gps_data.latitude,
            lon=gps_data.longitude,
            alt=gps_data.altitude,
            heading=0
        )

    def estimate_velocity(self) -> float:
        return self._odometer.read()
    
        # TODO
        # check if this is real or a Kalman Filter may be necessary to estimate v with imu data integration

    def estimate_ego_pose (self) -> MapPose:
               
        pose: WorldPose = self.__read_raw_world_data()
        if pose is None:
            return None
        return self._coordinate_converter.get_relative_map_pose(pose)
           
        # TODO - Make the real thing, using Kalman Filter for pose and heading estimation
        #return self._kalman_filter.estimate_pose(self._gps.read(), self._imu.read(), self._odometer.read())
        