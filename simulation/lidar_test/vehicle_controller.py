#! /usr/bin/python3
from carladriver import CarlaSimulation
import time
import cv2
import numpy as np

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
    #sim.reset()
    print ("summoning the EGO vehicle...")
    ego = sim.add_ego_vehicle(
        pos=[-90, 0, 3], 
        rotation=(0, 0, 0))
    
    lidar = ego.init_lidar(period_ms=100)

    camera = ego.init_rgb_bev_camera()

    time.sleep(1)
    frame = None
    while frame is None:
        frame, ts = camera.read()
        time.sleep(0.1)

    cv2.imwrite("bev.png", frame)


    for i in range(100):
        point_cloud, ts = lidar.read()
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        if data is not None:
            print(f"num points: {data.shape}")
        time.sleep(0.1)

    # while True:
    #     time.sleep(1)

    #lidar = ego.init_lidar()

    print("press enter to terminate")
    input()

    ego.destroy()
    
        



if __name__ == "__main__":
    main()