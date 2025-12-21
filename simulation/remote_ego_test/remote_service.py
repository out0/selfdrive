#! /usr/bin/python3
from carladriver import CarlaSimulation
from pydriveless import RemoteGPSServer, RemoteIMUServer, RemoteCameraServer
from pydriveless import RemoteEgoServer, EgoVehicle
import time

#
#  This script connects to a CARLA simulator, spawns an EGO vehicle, and initializes the remote services
#  for GPS, IMU, camera, and EGO vehicle control. The remote services use datalink to decouple communication via TCP/IP.
#
#

class LocalClient(EgoVehicle):
    def set_power(self, power_level: float) -> None:
        pass
    
    def set_brake(self, brake_level: float) -> None:
        pass
    
    def set_steering(self, angle: float) -> None:
        pass


def main():
    print ("connecting to the simulator...")
    sim = CarlaSimulation(
        town_name='Town07'
    )
    #sim.reset()
    print ("summoning the EGO vehicle...")
    ego = sim.add_ego_vehicle(
        pos=[-90, 0, 3], 
        rotation=(0, 0, 0))
    
    gps = ego.attach_gps_sensor(10)
    imu = ego.attach_imu_sensor(10)

    camera = ego.init_rgb_bev_camera()
    camera2 = ego.init_semantic_bev_camera()
    
    RemoteEgoServer(ego)
    RemoteIMUServer(imu, imu_period_ms=10)
    RemoteCameraServer(camera, period_ms=100)    
    RemoteCameraServer(camera2, period_ms=100, port=27111)    
    RemoteGPSServer(gps, gps_period_ms=10)

    # while True:
    #     time.sleep(1)

    #lidar = ego.init_lidar()

    print("press enter to terminate")
    input()

    ego.destroy()
    
        



if __name__ == "__main__":
    main()