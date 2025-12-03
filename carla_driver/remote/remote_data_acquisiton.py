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
    gps = RemoteGPSClient(port=22002)
    imu = RemoteIMUClient(port=22003)
    bev = RemoteCameraClient()

    while True:
        print ("Sensor data: ")
        print (" GPS: ", gps.read())
        print (" IMU: ", imu.read())
        frame, timestamp  = bev.read()
        if frame is not None:
            print (f" Camera: frame shape: {frame.shape} [{timestamp}]")
            cv2.imwrite(f"bev_frame.png", frame)
        else:
            print (" Camera: no frame received")
        time.sleep(0.5)


if __name__ == "__main__":
    main()