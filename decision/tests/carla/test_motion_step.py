import sys, os
sys.path.append("../../")
from ensemble import VehicleController, LocalPlannerType, PhysicalParameters, PlanningPipeline, Interpolator, MotionController, PlannerResultType
import time
from pydriveless import WorldPose, MapPose, Waypoint, CoordinateConverter, angle
from carla_test_utils import read_path, init_sim
import cv2
from carladriver import CarlaEgoVehicle, CarlaSimulation, CarlaSLAM, BevCameraSemantic
from pydriveless import Telemetry
import json

GPS_PERIOD_MS=100
IMU_PERIOD_MS=100

###
## Step by step tester
###
path: list[MapPose] = None
with open("driving_planned_path.json") as f:
    data = f.read()
    m = json.loads(data.replace("\n", ""))
    path = [MapPose.from_str(p) for p in m]

sim, ego, slam = init_sim()
time.sleep(2)
sim.show_path(path)

print (f"hit enter to drive the stored path with {len(path)} points")
input()

motion_controller = MotionController(
        period_ms=2,
        longitudinal_controller_period_ms=50,
        ego=ego,
        slam=slam,
        odometer=ego.get_odometer_sensor()
    )
motion_controller.start()
motion_controller.set_path(path, velocity=2)

while motion_controller.is_tracking():
    time.sleep(0.01)
    pass
#motion_controller.cancel()
motion_controller.brake()
input()

ego.destroy()