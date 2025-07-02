import sys
sys.path.append("../../../")
from typing import List
from model.world_pose import WorldPose
from utils.telemetry import Telemetry
import cv2
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters

# Estudar
# https://www.youtube.com/watch?v=_aqwJBx2NFk

import time

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

def execute_pre_process (seq: int) -> None:
    
    result = Telemetry.read_planning_result(seq)
    if result is None:
        print (f"no log found for plan #{seq}")
        return False
    
    bev = Telemetry.read_planning_bev(seq)

    og = OccupancyGrid(
                frame=bev,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )

    og.set_goal_vectorized(result.local_goal)
    f = og.export_free_areas(angle=0.0)
    cv2.imwrite(f"test_free_area_{seq}.png", f)
    
    # for i in range(-70, 90):
    #     f = og.export_free_areas(angle=i)
    #     cv2.imwrite(f"test_free_area_{seq}.png", f)
    #     time.sleep(0.250)
    #     print (f"angle: {i} deg")


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
    # if RUN_ALL:
    #     for i in range(1,1000):
    #         if not execute_plan(i): break
    #     return
    
    # while True:
    execute_pre_process(1)
    #execute_plan(17)
    # execute_plan(3)
    # execute_plan(4)
    # execute_plan(5)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))