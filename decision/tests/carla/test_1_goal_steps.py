import sys, os
sys.path.append("../../")
from ensemble import PhysicalParameters,  Ensemble, PlanningData, Overtaker
from ensemble import Interpolator, Overtaker, HybridAStar, BiRRTStar
from pydriveless import CoordinateConverter
import time
from pydriveless import WorldPose, MapPose, Waypoint,  angle
from test_utils import read_path, export_planning_response
import cv2
#from carladriver import CarlaEgoVehicle, CarlaSimulation, CarlaSLAM, BevCameraSemantic
from pydriveless import Telemetry, SearchFrame
import json
import numpy as np
import cProfile, timeit


GPS_PERIOD_MS=100
IMU_PERIOD_MS=100

###
## Step by step tester: Planner
###

#file = "log/timeout_planning"
#file = "log/invalid_planning"
file = "log/bev_0"

path = read_path("test_motion_controller_goal_points.dat")

planner_data: SearchFrame = None
with open(f"{file}.log", "r") as f:
    j = f.read()
    planner_data :PlanningData  = PlanningData.from_str(j)

path_pos = planner_data.seq

#frame = np.array(cv2.imread(f"{file}_bev.png"), dtype=np.float32)
frame = np.array(cv2.imread(f"log/bev_0.png"), dtype=np.float32)

planner_data.og().set_frame_data(frame)
planner_data.og().set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
planner_data.og().set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)

pos = MapPose.find_nearest_goal_pose(
            location=planner_data.ego_location(),
            poses=path,
            start=path_pos,
            max_hopping=5
        )   
hopping_distance_to_end = (path_pos - len(path))
if pos < 0:
    if hopping_distance_to_end > 1:
        print ("invalid self position")
        exit(0)
    else:
        print ("finish driving")
        exit(0)
        
if pos == len(path) - 1:
    g1, g2 = path[-1], None
else:
    g1, g2 = path[pos], path[pos + 1]

print (f"driving to g1 = {g1}, g2 = {g2}")
print (f"current pos: {planner_data.ego_location()}")
