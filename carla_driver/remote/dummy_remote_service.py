#! /usr/bin/python3
from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSServer, RemoteIMUServer, RemoteCameraServer
from pydriveless import RemoteEgoServer, GPS, GpsData, IMU, IMUData
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


def main():
    
    RemoteGPSServer(DummyGps(0), gps_period_ms=250, port=22002)
    RemoteIMUServer(DummyImu(0), imu_period_ms=10, port=22003)

    #bev = ego.init_rgb_bev_camera()
    #RemoteCameraServer(bev, period_ms=100)
    
    #RemoteEgoServer(ego)

    while True:
        time.sleep(1)

        



if __name__ == "__main__":
    main()