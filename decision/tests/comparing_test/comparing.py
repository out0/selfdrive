import sys, os
sys.path.append("../../")
import time
from ensemble import ReedsShepp, PhysicalParameters, Ensemble, PlanningData, PlanningResult
from pydriveless import WorldPose, MapPose, Waypoint,  angle
import cv2
#from carladriver import CarlaEgoVehicle, CarlaSimulation, CarlaSLAM, BevCameraSemantic
from pydriveless import Telemetry, SearchFrame, CoordinateConverter
import json
import numpy as np
import cProfile, timeit
import math

EGO_DIMENSIONS_PX=(20, 40)
ORIGIN=WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

###
## Comparing Ensemble and S-O-T-A LP
###

def frame_to_og(frame: np.ndarray) -> SearchFrame:
    
    f = np.zeros(frame.shape, dtype=np.float32)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
             f[i, j, 0] = 1.0 if frame[i, j, 0] == 255 else 0
             #f[i, j, 0] = 1.0

    og = SearchFrame(
     width=frame.shape[1],
     height=frame.shape[0],
     lower_bound=(start.x - EGO_DIMENSIONS_PX[0], start.z + EGO_DIMENSIONS_PX[1]),
     upper_bound=(start.x + EGO_DIMENSIONS_PX[0], start.z - EGO_DIMENSIONS_PX[1]))

    og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
    og.set_frame_data(f)
    return og


def mat_trans(x: int, z: int) -> np.ndarray:
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [x , z, 1]
    ])

def add_ego(frame, start: Waypoint) -> None:
    hh = EGO_DIMENSIONS_PX[1] // 2
    ww = EGO_DIMENSIONS_PX[0] // 2
    
    c = math.cos(start.heading.rad())
    s = math.sin(start.heading.rad())
    x = start.x
    z = start.z

    Mr = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0 , 0, 1]
    ])
    M = mat_trans(-x, -z) @ Mr @ mat_trans(x, z)
    
    for j in range(-hh, hh):
        for i in range(-ww, ww):
            p = np.array([x+i, z+j, 1]) @ M
            # zp = z + j
            # xp = x + i
            xp = int(p[0])
            zp = int(p[1])
            if xp < 0 or xp >= frame.shape[1]: continue
            if zp < 0 or zp >= frame.shape[0]: continue
            frame[zp, xp] = [255, 0, 0]

    draw_arrow(frame, x, z, start.heading.rad(), arrow_length=50)

def draw_arrow(frame: np.ndarray, x: int, z: int, heading_rad: float, arrow_length = 20):
    # Arrow end point
    end_x = int(x + arrow_length * math.sin(math.pi + heading_rad))
    end_z = int(z + arrow_length * math.cos(math.pi + heading_rad))

    # Draw the arrow shaft (line)
    cv2.line(frame, (x, z), (end_x, end_z), (0, 0, 255), thickness=2)
    # Draw arrow head (simple triangle)
    angle = math.atan2(end_z - z, end_x - x)
    arrow_head_size = 8
    for side in [-1, 1]:
        side_angle = angle + side * math.radians(25)
        hx = int(end_x - arrow_head_size * math.cos(side_angle))
        hy = int(end_z - arrow_head_size * math.sin(side_angle))
        cv2.line(frame, (end_x, end_z), (hx, hy), (0, 0, 255), thickness=2)

frame = np.array(cv2.imread(f"comparing_og.png"), dtype=np.uint8)

start = Waypoint(455, 263, angle.new_deg(0))
#goal =  Waypoint(143, 114, angle.new_deg(-70))
#goal =  Waypoint(121, 107, angle.new_deg(-180))
goal =  Waypoint(48, 261, angle.new_deg(-180))

conv = CoordinateConverter(
     origin=ORIGIN, 
     width=frame.shape[1], 
     height=frame.shape[0],
     perceptionHeightSize_m=frame.shape[0]*PhysicalParameters.OG_WIDTH_PX_TO_METERS_RATE,
     perceptionWidthSize_m=frame.shape[0]*PhysicalParameters.OG_HEIGHT_PX_TO_METERS_RATE)

og = frame_to_og(frame)
#og.process_distance_to_goal(goal.x, goal.z)
#og.process_safe_distance_zone(min_distance=(20, 40), compute_vectorized=True)

map_center_location = MapPose(0, 0, 0, angle.new_rad(0))
ego_location = conv.convert(map_center_location, start)
l0 = conv.convert(map_center_location, ego_location)
g1 = conv.convert(map_center_location, pose=goal)
g2 = None

planning_data = PlanningData(seq=1, og=og, ego_location=ego_location, start=start,
                             g1=g1, g2=g2, velocity=2.0, min_distance=(20, 40),
                             base_map_conversion_location=map_center_location)

planning_data.set_local_goal(goal)
planner = Ensemble(conv, max_exec_time_ms=-1)
planner.plan(planning_data)

while not planner.new_path_available():
     pass

ensemble_res = planner.get_result()

path_ensemble = ensemble_res.path

path = ReedsShepp(step=0.1, vehicle_length_m=PhysicalParameters.VEHICLE_LENGTH_M, max_steering_angle=PhysicalParameters.MAX_STEERING_ANGLE, speed=5.0)
_, _, x_list, y_list, pw = path.generation(
    start_pose=start,
    #goal_pose=(121, 107, angle.new_deg(-30).rad())
    goal_pose=goal
)

add_ego(frame, start)
for j in range(len(x_list)):
    z = int(y_list[j])
    x = int(x_list[j])
    if z < 0 or z >= frame.shape[0]: continue
    if x < 0 or x >= frame.shape[1]: continue
    frame[z, x, :] = [0, 255, 0]

for p in path_ensemble:
    frame[p.z, p.x, :] = [128, 0, 128]

print(f"Ensemble path result: {ensemble_res.planner_name}")

cv2.imwrite("debug.png", frame)

P = 1

