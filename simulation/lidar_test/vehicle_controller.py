#! /usr/bin/python3
from carladriver import CarlaSimulation
from pydriveless import CoordinateConverter, WorldPose, angle, MapPose
import time
import cv2
import numpy as np

#
#  This script connects to a CARLA simulator, spawns an EGO vehicle, and initializes the remote services
#  for GPS, IMU, camera, and EGO vehicle control. The remote services use datalink to decouple communication via TCP/IP.
#
#

OG_REAL_WIDTH: float = 34.641016151377535
OG_REAL_HEIGHT: float = 34.641016151377535
OG_WIDTH: int = 256
OG_HEIGHT: int = 256
ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0.0, angle.new_rad(0))
    
OG_WIDTH_PX_TO_METERS_RATE: float = OG_REAL_WIDTH / OG_WIDTH
OG_HEIGHT_PX_TO_METERS_RATE: float = OG_REAL_HEIGHT / OG_HEIGHT

def main():
    print ("connecting to the simulator...")
    sim = CarlaSimulation(
        town_name='Town07'
    )
    #sim.reset()
    print ("summoning the EGO vehicle...")
    ego_location = MapPose(-90.0, 0, 3, heading=angle.new_deg(0), reversed=False)
    pos = [-90, 0, 3]
    ego = sim.add_ego_vehicle(
        pos=pos, 
        rotation=(0, 0, 0))
    
    lidar = ego.init_lidar(period_ms=100)

    camera = ego.init_rgb_bev_camera()

    time.sleep(1)
    frame = None
    while frame is None:
        frame, ts = camera.read()
        time.sleep(0.1)

    cv2.imwrite("bev.png", frame)

    conv = CoordinateConverter(ORIGIN, OG_WIDTH, OG_HEIGHT, OG_REAL_WIDTH, OG_REAL_HEIGHT)
    

    for i in range(10):
        point_cloud, ts = lidar.read()
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        num_rows = int(data.shape[0] / 4)
        data = np.reshape(data, (num_rows, 4))
        for i in range(num_rows):
            p = data[i][0:3] + [0, 0, 0]
            map_p = MapPose(p[0], p[1], p[2])
            local_p = conv.convert(location=ego_location, pose=map_p)
            frame[local_p.z, local_p.x, :] = [255, 255, 255]
            
            #sim.show_coordinate(pose=p, color=[255, 255, 255])
        if data is not None:
            print(f"num points: {data.shape}")
        time.sleep(0.1)

    cv2.imwrite("bev.png", frame)

    # while True:
    #     time.sleep(1)

    #lidar = ego.init_lidar()

    print("press enter to terminate")
    input()

    ego.destroy()
    
        



if __name__ == "__main__":
    main()