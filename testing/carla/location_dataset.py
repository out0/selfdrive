import sys
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar

from data_logger import DataLogger

client = CarlaClient(town='Town07')
ego = CarlaEgoCar(client)
ego.set_pose(-100, 0, 0, 0)
ego.set_brake(1.0)

logger = DataLogger(10, ego, "location.log")

print ("Press <enter> to start recording")
input()
logger.start()
ego.set_autopilot(True)

print ("Press <enter> to stop recording and terminate")
input()
logger.destroy()


