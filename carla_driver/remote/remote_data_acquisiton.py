#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteGPSClient, RemoteIMUClient, RemoteCameraClient
import numpy as np
import time
import cv2
import faulthandler
faulthandler.enable()

from pydatalink import Datalink
import numpy as np
from pydriveless import GPS, GpsData
import time
import threading


class RemoteGPSClient(GPS):
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _gps: GPS

    def __init__(self, host: str = "127.0.0.1", port: int = 22002):
        self._running = True
        self._last_gps_data = None
        self._data_link = Datalink(host=host, port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._read_sensor_data)
        self._data_thread.start()
    
    def __del__(self):
        self._running = False
        self._data_thread.join()
        del self._data_link
    
    def _read_sensor_data(self) -> None:
        while self._running:
            if not self._data_link.is_ready():
                print ("GPS server: data link not ready")
                while not self._data_link.is_ready():
                    time.sleep(0.01)
                print ("GPS server: Online")
                continue
            
            if not self._data_link.has_data():
                time.sleep(0.01)
                continue

            data, size, timestamp = self._data_link.read_np(shape=(4,), dtype=np.float32)
            if size == 0:
                continue

            self._last_gps_data = GpsData(
                lat=data[0],
                lon=data[1],
                alt=data[2],
                timestamp=timestamp,
                valid=True)
    
    def read(self) -> GpsData:
        return self._last_gps_data

def main():
    gps = RemoteGPSClient(port=22002)
    imu = RemoteIMUClient(port=22003)

    #bev = RemoteCameraClient()
    i = 0
    while True:
        print ("Sensor data: ")
        print (" GPS: ", gps.read())
        print (" IMU: ", imu.read())
     #   frame = bev.read()
        # if frame is not None:
        #     print (" Camera frame shape: ", frame.shape)
        #     cv2.imwrite(f"bev_frame.png", frame)
        time.sleep(0.5)


if __name__ == "__main__":
    main()