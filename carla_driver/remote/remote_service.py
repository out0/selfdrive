#! /usr/bin/python3
from carladriver import CarlaSimulation
from pydriveless import RemoteGPSServer, RemoteIMUServer, RemoteCameraServer
from pydriveless import RemoteEgoServer
import time

#
#  This script connects to a CARLA simulator, spawns an EGO vehicle, and initializes the remote services
#  for GPS, IMU, camera, and EGO vehicle control. The remote services use datalink to decouple communication via TCP/IP.
#
#


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
    
    gps = ego.attach_gps_sensor(250)
    imu = ego.attach_imu_sensor(10)
    camera = ego.init_rgb_bev_camera()
    
    RemoteGPSServer(gps, gps_period_ms=250)
    RemoteIMUServer(imu, imu_period_ms=10)
    RemoteCameraServer(camera, period_ms=100)    
    RemoteEgoServer(ego)

    while True:
        time.sleep(1)

        



if __name__ == "__main__":
    main()