#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSClient, RemoteIMUClient, RemoteCameraClient
import numpy as np
import time
import cv2

#
#  This script reads data from remote sensors (GPS, IMU, camera) connected to a CARLA simulator via datalink.
#  see remote_service.py for the server side.
#
#

def main():
    gps = RemoteGPSClient()
    imu = RemoteIMUClient()
    camera = RemoteCameraClient()
    camera2 = RemoteCameraClient(port=27111)

    while True:
        print ("Sensor data: ")
        print (" GPS: ", gps.read())
        print (" IMU: ", imu.read())
        frame, timestamp  = camera.read()
        if frame is not None and timestamp > 0 and len(frame.shape) > 0 and frame.shape[0] != 0:
            print (f" Camera: frame shape: {frame.shape} [{timestamp}]")
            cv2.imwrite(f"bev_frame.png", frame)
        else:
            print (" Camera: no frame received")

        frame, timestamp  = camera2.read()
        if frame is not None and timestamp > 0 and len(frame.shape) > 0 and frame.shape[0] != 0:
            print (f" Camera2: frame shape: {frame.shape} [{timestamp}]")
            cv2.imwrite(f"bev2_frame.png", frame)
        else:
            print (" Camera2: no frame received")


        time.sleep(0.5)


if __name__ == "__main__":
    main()