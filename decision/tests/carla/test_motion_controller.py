import sys, os
sys.path.append("../../")
from motion.motion_controller import MotionController
import time
from carladriver import CarlaSimulation, CarlaEgoVehicle, CarlaSLAM, CarlaOdometer
from pydriveless import MapPose

sim = CarlaSimulation(town_name='Town07')
ego = sim.add_ego_vehicle(pos=[-90, 0, 3])
ego.set_brake(1.0)

def on_finished_motion(controller: MotionController):
    print("the mission has finished succesfully")
    controller.cancel()
    controller.brake()

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
    slam=CarlaSLAM(ego.get_carla_obj()),
    odometer=CarlaOdometer(ego.get_carla_obj()),
    on_finished_motion=on_finished_motion
)

path = read_path("test_motion_controller_goal_points.dat")

sim.show_path(path)

init_pose = path[0]
ego.set_pose(init_pose)
time.sleep(3)

controller.set_path(path, velocity=5.0)
controller.start()

print ("Press <enter> to stop")
input()

controller.destroy()
ego.destroy()
