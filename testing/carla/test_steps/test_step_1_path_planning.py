import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.world_pose import WorldPose
from slam.slam import SLAM
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from utils.logging import Telemetry
from testing.unittests.test_utils import PlannerTestOutput

import time

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = -1
PLANNER_TYPE = LocalPlannerType.HybridAStar

def execute_plan (seq: int) -> None:
    coord = CoordinateConverter(COORD_ORIGIN)

    local_planner = LocalPlanner(
        plan_timeout_ms=PLAN_TIMEOUT,
        local_planner_type=PLANNER_TYPE,
        map_coordinate_converter=coord
    )

    result = Telemetry.read_planning_result(seq)
    bev = Telemetry.read_planning_bev(seq)

    data = PlanningData(
        bev = bev,
        ego_location=result.ego_location,
        goal=result.map_goal,
        next_goal=result.map_next_goal,
        velocity=5.0
    )

    local_planner.plan(data)

    while local_planner.is_planning():
        time.sleep(0.01)

    res = local_planner.get_result()

    outp = PlannerTestOutput(
        frame=data.og.get_color_frame(),
        convert_to_gray=True
    )

    outp.add_path(res.path)
    outp.write(f"test_output_{seq}.png")

def main(argc: int, argv: List[str]) -> int:
    execute_plan(4)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))