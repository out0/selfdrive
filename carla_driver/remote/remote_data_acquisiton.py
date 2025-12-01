#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSClient, RemoteIMUClient, RemoteCameraClient
import numpy as np
import time
import cv2

def main():


    gps = RemoteGPSClient()
    imu = RemoteIMUClient()
    bev = RemoteCameraClient()
    i = 0
    while True:
        print ("Sensor data: ")
        print (" GPS: ", gps.read())
        print (" IMU: ", imu.read())
        frame = bev.read()
        if frame is not None:
            print (" Camera frame shape: ", frame.shape)
            cv2.imwrite(f"bev_frame.png", frame)
        time.sleep(0.5)


if __name__ == "__main__":
    main()