#
# This script reads telemetry data from vehicle_controller, saves frame data, and displays gps/imu data 
#


#
# This script intializes a Ego vehicle in Carla and attaches a Telemetry logger to direct-access vehicle sensor data
#

from pydriveless import TelemetryReader
import numpy as np
import cv2

class DataReader(TelemetryReader):
    def _on_log_received(self, id, data, timestamp):
        if id == "gps":
            print(f"[GPS] {data}")
        elif id == "imu":
            print(f"[IMU] {data}")
        elif id == "camera":
            new_arr = np.frombuffer(data, dtype=np.uint8)
            print(f"[FRAME] {new_arr.shape}")
            cv2.imwrite("bev.png", new_arr.reshape(256, 256, 3))


        return super()._on_log_received(id, data, timestamp)

def main():
    reader = DataReader(host="127.0.0.1", port=30000)
    input()


if __name__ == "__main__":
    main()