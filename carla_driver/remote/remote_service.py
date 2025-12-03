#! /usr/bin/python3
from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSServer, RemoteIMUServer, RemoteCameraServer
from pydriveless import RemoteEgoServer
import numpy as np
import time
#import faulthandler
#faulthandler.enable()

CMD_SIZE = 10
KEEP_ALIVE_RESPONSE = np.zeros(CMD_SIZE, dtype=np.float32)


def main():
    print ("connecting to the simulator...")
    sim = CarlaSimulation(
        town_name='Town07'
    )
    sim.reset()
    print ("summoning the EGO vehicle...")
    ego = sim.add_ego_vehicle(
        pos=[-90, 0, 3], 
        rotation=(0, 0, 0))
    
    RemoteGPSServer(ego.attach_gps_sensor(250), gps_period_ms=250, port=22002)
    RemoteIMUServer(ego.attach_imu_sensor(10), imu_period_ms=10, port=22003)

    bev = ego.init_rgb_bev_camera()
    RemoteCameraServer(bev, period_ms=100)    
    RemoteEgoServer(ego)

    while True:
        time.sleep(1)

        



if __name__ == "__main__":
    main()