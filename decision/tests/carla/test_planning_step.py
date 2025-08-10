import sys, os
sys.path.append("../../")
from ensemble import PhysicalParameters,  Ensemble, PlanningData, Overtaker
from ensemble import Interpolator, Overtaker, HybridAStar, BiRRTStar
from ensemble import CoordinateConverter
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

file = "log/timeout_planning"

planner_data: SearchFrame = None
with open(f"{file}.log", "r") as f:
    j = f.read()
    planner_data = PlanningData.from_str(j)

frame = np.array(cv2.imread(f"{file}_bev.png"), dtype=np.float32)

planner_data.og().set_frame_data(frame)
planner_data.og().set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
planner_data.og().set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)


og: SearchFrame = planner_data.og()
local_goal = planner_data.local_goal()

og.process_safe_distance_zone((PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX), True)
og.process_distance_to_goal(local_goal.x, local_goal.z)

cv2.imwrite("test_output_a.png", og.get_color_frame())

ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0.0, angle.new_rad(0))
conv = CoordinateConverter(
    origin=ORIGIN, 
    width=PhysicalParameters.OG_WIDTH, 
    height=PhysicalParameters.OG_HEIGHT, 
    perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT,
    perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH)

export_planning_response("planning_response.png", planner_data, None)
planner = Ensemble(conv, -1)
#planner = Overtaker(-1)

#planner = HybridAStar(conv, -1, dist_to_target_tolerance=20)

start_time = time.time()
planner.plan(planner_data)
while not planner.new_path_available():
    time.sleep(0.1)
execution_time = time.time() - start_time
res = planner.get_result()
print(f"planner execution time: {1000*execution_time:.6f} ms [choosen: {res.planner_name}]")
#planner = Interpolator(conv, -1)

#cProfile.run("planner.plan(planner_data, True)")
# execution_time = timeit.timeit(lambda: planner.plan(planner_data, True), number=1)
# print(f"planner.plan execution time: {execution_time:.6f} seconds")


export_planning_response("planning_response.png", planner_data, res)
