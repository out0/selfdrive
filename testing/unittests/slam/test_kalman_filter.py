import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest

from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from model.world_pose import WorldPose
from data.coordinate_converter import CoordinateConverter
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
        
        dict = json.loads(raw[4:])
        
        if raw[0:3] == "gps":
            return (
                GpsData.from_str(dict['gps']),
                None,
                MapPose.from_str(dict['pose']),
                float(dict['velocity']),
                float(dict['timestamp'])
            )
        else:     
            return (
                None,
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
            if (imu != None):
                filter.calibrate()
                return i
            filter.add_calibration_gnss_data(location)
    
    def assertEqualLocation(self, l1: MapPose, l2: MapPose, v: float):
        print (f"error in x: ({l2.x - l1.x})")
        print (f"error in y: ({l2.y - l1.y})")
        print (f"error in heading: ({l2.heading - l1.heading})")
        print (f"velocity: {v}")
        #self.assertAlmostEqual(l1.x, l2.x, places=0)
        #self.assertAlmostEqual(l1.y, l2.y, places=0)
        #self.assertAlmostEqual(l1.heading, l2.heading, places=1)
    
    def test_efk_dev(self):
        f = open("location.log", "r")
        
        dt_imu = 50/1000
        dt_gps = 500/1000
        
        filter = ExtendedKalmanFilter()
        
        lines = f.readlines()
        init = self.calibrate(filter, lines)
        
        for i in range(init, len(lines)):
            _, _, _, vel, _ = self.__convert_data(lines[i])
            if vel >= 1: 
                init = i
                break
            
        #first_in_move 
        gps, imu, expected_pose, vel, dt = self.__convert_data(lines[init])
        filter.predict_state_with_imu(imu, dt_imu)
        
        
        for i in range(init+1, len(lines)):
            gps, imu, expected_pose, vel, dt = self.__convert_data(lines[i])
            if gps is not None:
                filter.correct_state_with_gnss(gps)
                
    
    
    def test_ekf(self):
        return
        f = open("location.log", "r")
        
        dt_imu = 50/1000
        dt_gps = 500/1000
        
        filter = ExtendedKalmanFilter()
        
        lines = f.readlines()
        init = self.calibrate(filter, lines)
        
        location_0, _, pose_0, vel, _ = self.__convert_data(lines[init - 1])
        
        pred_pose = filter.get_location()
        self.assertEqualLocation(pose_0, pred_pose, vel)
        
            
        for i in range(init, len(lines)):
            gps, imu, expected_pose, vel, dt = self.__convert_data(lines[i])
            
            if gps is not None:
                filter.correct_state_with_gnss(gps, expected_pose.heading)
            
            if imu is not None:
                filter.predict_state_with_imu(imu, dt_imu)
            
            pred_pose = filter.get_location()
            self.assertEqualLocation(expected_pose, pred_pose, vel)
        
        p = 1

if __name__ == "__main__":
    unittest.main()
