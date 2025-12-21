#! /usr/bin/python3
from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSServer, RemoteIMUServer, RemoteCameraServer
from pydriveless import RemoteEgoServer, GPS, GpsData, IMU, IMUData, Camera, EgoVehicle
import numpy as np
import time
#import faulthandler
#faulthandler.enable()

CMD_SIZE = 10
KEEP_ALIVE_RESPONSE = np.zeros(CMD_SIZE, dtype=np.float32)

class DummyGps(GPS):
    def read(self) -> GpsData:
        return GpsData(
            lat=37.7749,
            lon=-122.4194,
            alt=15.0,
            valid=True,
            timestamp=time.time()
        )
class DummyImu(IMU):
    def read(self) -> IMUData:
        return IMUData(
            accel_x=1.0,
            accel_y=2.0,
            accel_z=9.81,
            compass=3.0,
            gyro_x=4.0,
            gyro_y=5.0,
            gyro_z=6.0,
            valid=True,
            timestamp=time.time()
        )

class DummyCamera(Camera):
    def __init__(self, width, height, fov = 120, fps = 30):
        super().__init__(width, height, fov, fps)

    def read(self) -> tuple[np.ndarray, float]:
        return np.full((self.height(), self.width(), 3), 255, dtype=np.uint8), time.time()

class DummyEgo(EgoVehicle):
        
    def set_power(self, power_level: float) -> None:
        print (f"power_level = {power_level}")
    
    def set_brake(self, brake_level: float) -> None:
        print (f"brake_level = {brake_level}")
    
    def set_steering(self, angle: float) -> None:
        print (f"steering angle = {angle}")

def main():
    
    RemoteGPSServer(DummyGps(0), gps_period_ms=250)
    RemoteIMUServer(DummyImu(0), imu_period_ms=10)

    #bev = ego.init_rgb_bev_camera()
    RemoteCameraServer(DummyCamera(256, 256), period_ms=100)
    
    ego = DummyEgo()
    RemoteEgoServer(ego)

    while True:
        time.sleep(1)

        



if __name__ == "__main__":
    main()