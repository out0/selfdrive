import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from slam.slam import SLAM
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from utils.logging import Telemetry
from testing.test_utils import PlannerTestOutput
from planner.goal_point_discover import GoalPointDiscover
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
import math

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = -1
PLANNER_TYPE = LocalPlannerType.Overtaker


def show_planned_location_relative_to_projection(
            file: str, 
            og: OccupancyGrid, 
            coord: CoordinateConverter,
            ego_location: MapPose, 
            local_start: Waypoint,
            goal: MapPose, 
            result_goal: Waypoint) -> None:
    
    outp = PlannerTestOutput(frame=og.get_color_frame())
    
    projected_goal = coord.clip(coord.convert_map_to_waypoint(ego_location, goal))
    
    print(f"projected local goal: ({projected_goal.x}, {projected_goal.z})")
    print(f"chosen local goal: ({result_goal.x}, {result_goal.z})")
    
    h = Waypoint.compute_heading(local_start, result_goal)
    print(f"direct heading:{h:.4} degrees")
    print(f"chosen heading: {result_goal.heading:} degrees")
    
    if projected_goal.x == result_goal.x and projected_goal.z == result_goal.z:
        outp.add_point(projected_goal, color=[0, 255, 0])
    else:
        outp.add_point(projected_goal, color=[255, 0, 0])
        outp.add_point(result_goal, color=[0, 0, 255])

    outp.draw_vector(Waypoint(128,128), result_goal.heading,  size=40)
    
    outp.write(file)
    

def execute_plan (seq: int) -> None:
    coord = CoordinateConverter(COORD_ORIGIN)
    local_goal_discover = GoalPointDiscover(coord)

    result = Telemetry.read_planning_result(seq)
    bev = Telemetry.read_planning_bev(seq)

    og = OccupancyGrid(
                frame=bev,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )

    res = local_goal_discover.find_goal(
        og=og,
        current_pose=result.ego_location,
        goal_pose=result.map_goal,
        next_goal_pose=result.map_next_goal
    )
    
    if res is None:
        print(f"no goal was found for seq: {seq}")
        return
    
    show_planned_location_relative_to_projection(
        file=f"goal_point_res_{seq}.png",
        og=og,
        coord=coord,
        local_start=result.local_start,
        ego_location=result.ego_location,
        goal=result.map_goal,
        result_goal=res.goal)


def main(argc: int, argv: List[str]) -> int:
    for i in range(1,25):
         execute_plan(i)
    
    # execute_plan(11)
    # execute_plan(3)
    # execute_plan(4)
    # execute_plan(5)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))