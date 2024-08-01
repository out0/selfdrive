import sys, os
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.carla_slam import CarlaSLAM
from carlasim.sensors.data_sensors import *
from model.map_pose import MapPose
from model.sensor_data import *
from motion.motion_controller import MotionController
from carlasim.debug.carla_debug import CarlaDebug
import time

client = CarlaClient(town='Town07')
ego = CarlaEgoCar(client)
ego.set_pose(-100, 0, 0, 0)
ego.set_brake(1.0)

def on_finished_motion():
    print("the mission has finished succesfully")

def read_path (file: str) -> list[MapPose]:
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    path = []
    for l in lines:
        p = l.replace("\n", "").split(';')
        path.append(MapPose(x=float(p[0]), y=float(p[1]), z=float(p[2]), heading=float(p[3])))
    return path
        

controller = MotionController(
    period_ms=2,
    longitudinal_controller_period_ms=50,
    ego=ego,
    slam=CarlaSLAM(ego),
    desired_speed=5.0,
    on_finished_motion=lambda p: p.brake()
)

path = read_path("test_motion_controller_goal_points.dat")

CarlaDebug(client).show_global_path(path)

time.sleep(3)

controller.set_path(path)
controller.start()

print ("Press <enter> to stop")
input()

controller.destroy()
ego.destroy()
