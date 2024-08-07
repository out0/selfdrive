import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest

from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from slam.slam import SLAM
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

class TestSLAM(unittest.TestCase):
    
    def test_non_ekf(self):
        gps = DummyGps()
        gps.set_from_str("11.1|11.2|11.3")

        imu = DummyIMU()
        imu.set_from_str("1.1|1.1|1.1|1.1|1.1|1.1|50")
        odom = DummyOdometer()
        
        slam = SLAM(gps, imu, odom)

        slam.initialize()

        pose = slam.estimate_ego_pose()
        
        self.assertEqual(pose.x, 0)
        self.assertEqual(pose.y, 0)
        self.assertEqual(pose.z, 0)
        self.assertEqual(pose.heading, -40)
    
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
    
    
    def test_ekf(self):
        f = open("location.log", "r")
       
        location, imu, pose, vel, dt = self.__convert_data(f.readline())
        
        noise_gps = np.random.normal(0, 0.0001, (3,1))
        location.latitude += noise_gps[0]
        location.longitude += noise_gps[1]
        location.altitude += noise_gps[2]
        noise_imu = np.random.normal(0, 2, (7,1)) 
        
        
        p = 1

if __name__ == "__main__":
    unittest.main()
