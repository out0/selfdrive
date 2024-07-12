import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest

from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from slam.slam import SLAM


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
        
        slam = SLAM(gps, None, None)
        pose = slam.estimate_ego_pose()
        
        self.assertEqual(pose.lat, 11.1)
        self.assertEqual(pose.lon, 11.2)
        self.assertEqual(pose.alt, 11.3)
    

if __name__ == "__main__":
    unittest.main()
