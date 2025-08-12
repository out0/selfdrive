import math
from pydriveless import WorldPose, MapPose, Waypoint, CoordinateConverter, PI
from pydriveless import SearchFrame, angle
from ensemble import PlanningData, PlanningResult, PlannerResultType, PhysicalParameters, HybridAStar
import numpy as np
import math
import cv2
import time
#
# Test suit
#
EGO_DIMENSIONS_PX=(20, 40)
ORIGIN=WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

def frame_to_og(frame: np.ndarray, start: Waypoint) -> SearchFrame:   
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

    
    og.set_class_colors(np.array([[0,0,0], [255,255,255]]))
    og.set_class_costs(np.array([[-1.0], [0.0]]))
    #og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
    og.set_frame_data(f)
    return og

def exec_test():
    
    start = Waypoint(455, 263, angle.new_deg(0))
    #goal =  Waypoint(48, 261, angle.new_deg(-180))
    goal =  Waypoint(48, 261, angle.new_deg(0))
    #goal =  Waypoint(207, 117, angle.new_deg(-180))
    
    frame  = np.array(cv2.imread("comparing_og.png"))
    og = frame_to_og(frame, start)

    conv = CoordinateConverter(
        origin=ORIGIN, 
        width=frame.shape[1], 
        height=frame.shape[0],
        perceptionHeightSize_m=frame.shape[0]*PhysicalParameters.OG_WIDTH_PX_TO_METERS_RATE,
        perceptionWidthSize_m=frame.shape[0]*PhysicalParameters.OG_HEIGHT_PX_TO_METERS_RATE)
    
    map_center_location = MapPose(0, 0, 0, angle.new_rad(0))
    ego_location = conv.convert(map_center_location, start)
    l0 = conv.convert(map_center_location, ego_location)
    g1 = conv.convert(map_center_location, pose=goal)
    g2 = None

    planning_data = PlanningData(seq=1, og=og, ego_location=ego_location, start=start,
                                g1=g1, g2=g2, velocity=2.0, min_distance=(20, 40),
                                base_map_conversion_location=map_center_location)

    planning_data.set_local_goal(goal)

    og.process_distance_to_goal(goal.x, goal.z)
    og.process_safe_distance_zone(EGO_DIMENSIONS_PX, True)

    planner = HybridAStar(conv, max_exec_time_ms=-1, veh_dims=EGO_DIMENSIONS_PX)

    start_time = time.time()
    planner.plan(planning_data)
    while not planner.new_path_available():
        time.sleep(0.1)
    execution_time = time.time() - start_time
    res = planner.get_result()
    print(f"planner execution time: {1000*execution_time:.6f} ms [choosen: {res.planner_name}]")

    path_ensemble = res.path

    for p in path_ensemble:
        frame[p.z, p.x, :] = [128, 0, 128]

    cv2.imwrite("debug.png", frame) 


if __name__ == "__main__":
    exec_test()