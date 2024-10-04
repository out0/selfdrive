import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from planner.local_planner.local_planner import LocalPlannerType
from data.coordinate_converter import CoordinateConverter
from utils.logging import Telemetry
from testing.test_utils import PlannerTestOutput
from planner.goal_point_discover import GoalPointDiscover
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
from utils.time_recorder import ExecutionTimeRecorder
import time

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = -1
PLANNER_TYPE = LocalPlannerType.Overtaker


def execute_plan (seq: int) -> bool:
    coord = CoordinateConverter(COORD_ORIGIN)

    result = Telemetry.read_planning_result(seq)
    if result is None:
        return False
    
    bev = Telemetry.read_planning_bev(seq)

    og = OccupancyGrid(
                frame=bev,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )

    goal_candidate_waypoing = coord.convert_map_to_waypoint(result.ego_location, result.map_goal)

    ExecutionTimeRecorder.initialize(file="./measurements.log")
    ExecutionTimeRecorder.start(f"set_vect_goal_{seq}")    
    og.set_goal(goal_candidate_waypoing)
    ExecutionTimeRecorder.stop(f"set_vect_goal_{seq}")
    return True

RUN_ALL = True
#RUN_ALL = False

def main(argc: int, argv: List[str]) -> int:
    
    if RUN_ALL:
        for i in range(1,1000):
            if not execute_plan(i): break
        return
    
    execute_plan(1)
    #execute_plan(17)
    # execute_plan(3)
    # execute_plan(4)
    # execute_plan(5)
    
    input()    
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))