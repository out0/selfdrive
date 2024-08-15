import sys, time
sys.path.append("../../../")
sys.path.append("../../")
from model.sensor_data import GpsData, IMUData
import unittest, math

from data.ego_location import EgoLocation
from model.ego_car import EgoCar
from model.sensors.odometer import *
from model.sensors.gps import *
from model.sensors.imu import *

class DummyOdometer(Odometer):   
    def read(self) -> float:
        return 12.2

class DummyGps(GPS):    
    def read(self) -> GpsData:
        return GpsData(
            10.1,
            -12.2,
            123.1
        )

class DummyIMU(IMU):
    def read(self) -> IMUData:
        p = IMUData()
        p.heading = 94.3
        return p

class DummyEgoCar(EgoCar):
    def get_odometer(self) -> Odometer:
        return DummyOdometer()

    def get_gps(self) -> GPS:
        return DummyGps()

    def get_imu(self) -> IMU:
        return DummyIMU()

class TestEgoLocation(unittest.TestCase):

    def test_read_ego_pose(self):
        ego_location = EgoLocation(DummyEgoCar())
        pose = ego_location.estimate_ego_pose()
        v = ego_location.ego_velocity()
        
        self.assertAlmostEqual(v, 12.2, 4)
        self.assertAlmostEqual(pose.lat, 10.1, 4)
        self.assertAlmostEqual(pose.lon, -12.2, 4)
        self.assertAlmostEqual(pose.alt,123.1, 4)
        self.assertAlmostEqual(pose.heading, 94.3, 4)
        

if __name__ == "__main__":
    unittest.main()
