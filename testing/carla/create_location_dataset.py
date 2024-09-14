import sys, os
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
from data_logger import DataLogger


client = CarlaClient(town='Town07')
ego = CarlaEgoCar(client)
ego.set_pose(-100, 0, 0, 0)
ego.set_brake(1.0)

logger = DataLogger(ego=ego, 
                    log_file="location.log",
                    gps_period_ms=500,
                    imu_period_ms=50,
                    gps_calibration_time_ms=8000)

print ("Press <enter> to start recording")
input()
logger.start()


print ("storing calibration data")
while not logger.has_calibration_data():
    time.sleep(0.01)
    continue

print ("calibration data done, moving...")
ego.set_autopilot(True)

print ("Press <enter> to stop recording and terminate")
input()
logger.destroy()


