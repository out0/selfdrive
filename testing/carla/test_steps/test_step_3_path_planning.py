import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.world_pose import WorldPose
from slam.slam import SLAM
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from utils.telemetry import Telemetry
from testing.test_utils import PlannerTestOutput
import cv2

# Estudar
# https://www.youtube.com/watch?v=_aqwJBx2NFk

import time

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

#PLAN_TIMEOUT = 60000
PLAN_TIMEOUT = -1
PLANNER_TYPE = LocalPlannerType.FastRRT
#PLANNER_TYPE = LocalPlannerType.Overtaker


def execute_plan (seq: int) -> None:
    coord = CoordinateConverter(COORD_ORIGIN)

    local_planner = LocalPlanner(
        plan_timeout_ms=PLAN_TIMEOUT,
        local_planner_type=PLANNER_TYPE,
        map_coordinate_converter=coord
    )

    result = Telemetry.read_planning_result(seq)
    if result is None:
        print (f"no log found for plan #{seq}")
        return False
    
    unseg_bev = Telemetry.read_pre_planning_bev(seq)
    
    bev = Telemetry.read_planning_bev(seq)

    data = PlanningData(
        unseg_bev=unseg_bev,
        bev = bev,
        ego_location_ubev=result.ego_location,
        ego_location=result.ego_location,
        goal=result.map_goal,
        next_goal=result.map_next_goal,
        velocity=1.0
    )
    
    # f = data.og.get_color_frame()
    # cv2.imwrite("test_output.png", f)

    local_planner.plan(data)

    while local_planner.is_planning():
        time.sleep(0.01)

    res = local_planner.get_result()
    
    outp = PlannerTestOutput(
        frame=data.og.get_color_frame(),
        convert_to_gray=True
    )
    
    if res.result_type == PlannerResultType.VALID:
        print (f"[{seq}] Valid plan for the selected local planner {res.planner_name} with {len(res.path)} points")
    elif res.result_type == PlannerResultType.TOO_CLOSE:
        print (f"[{seq}] Ignoring path because its too close")
        outp.add_point(res.local_goal, color=[0,0,255])
        outp.write(f"test_output_{seq}.png")
        return True
    else:
        print (f"[{seq}] invalid plan for the selected local planner {res.planner_name}")
        outp.add_point(res.local_goal, color=[0,0,255])
        outp.write(f"test_output_{seq}.png")
        return True

    

    outp.add_path(res.path)
    outp.add_point(res.local_goal, color=[0,0,255])
    outp.write(f"test_output.png")
    return True

#RUN_ALL = True
RUN_ALL = False

def main(argc: int, argv: List[str]) -> int:
    
    # bev = Telemetry.read_planning_bev(1)
    
    # h, w = 90, 128
    
    # for i in range(0, 10):
    #     for j in range(-10, 30):
    #         bev[i+h, j+w] = bev[128, 128]
    #         pass
    
    # cv2.imwrite('../planning_data/bev_1a.png', bev)
    
    #execute_plan(1)
    #return
    if RUN_ALL:
        for i in range(1,1000):
            if not execute_plan(i): break
        return
    
    while True:
        execute_plan(1)
    #execute_plan(17)
    # execute_plan(3)
    # execute_plan(4)
    # execute_plan(5)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))