import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest

from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from slam.slam import SLAM
from slam.kalman_filter import ExtendedKalmanFilter
from model.map_pose import MapPose
import json, numpy as np


class DummyGps(GPS):
    _data: GpsData
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set_from_str(self, raw_str: str) -> None:
        self._data = GpsData.from_str(raw_str)
    
    def read(self) -> GpsData:
        return self._data
    
class DummyIMU(GPS):
    _data: IMUData
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set_from_str(self, raw_str: str) -> None:
        self._data = IMUData.from_str(raw_str)
    
    def read(self) -> IMUData:
        return self._data
    
class DummyOdometer(Odometer):
    _data: float
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set(self, val: float) -> None:
        self._data = val
    
    def read(self) -> float:
        return self._data

class TestKalmanFilter(unittest.TestCase):
    
    
    def __convert_data (self, raw: str) -> tuple[GpsData, IMUData, MapPose, float, float]:
        dict = json.loads(raw)
        return (
            GpsData.from_str(dict['gps']),
            IMUData.from_str(dict['imu']),
            MapPose.from_str(dict['pose']),
            float(dict['velocity']),
            float(dict['timestamp'])
        )
        
    def add_noise(arr: np.ndarray, std_dev: float) -> np.ndarray:
        noise = np.random.normal(0, std_dev, arr.shape) 
        return arr + noise
    
    
    def calibrate(self, filter: ExtendedKalmanFilter,  lines: list[str]) -> int:
        for i in range(len(lines)):
            location, imu, pose, vel, dt = self.__convert_data(lines[i])
            if (vel != 0):
                filter.calibrate()
                return i
            filter.add_calibration_gnss_data(pose)
    
    def test_ekf(self):
        f = open("location.log", "r")
        filter = ExtendedKalmanFilter()
        
        lines = f.readlines()
        init = self.calibrate(filter, lines)
        
        location_0, _, pose_0, _, _ = self.__convert_data(lines[init - 1])
        
            
        for i in range(init, len(lines)):
            location, imu, pose, vel, dt = self.__convert_data(lines[i])
            filter.predict_state_with_imu(imu, dt)
            filter.correct_state_with_gnss(pose)
            l = filter.get_location()            
            last_dt = dt
        
        p = 1

if __name__ == "__main__":
    unittest.main()
