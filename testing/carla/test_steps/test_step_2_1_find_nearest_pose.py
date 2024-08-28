import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.world_pose import WorldPose
from data.coordinate_converter import CoordinateConverter
from utils.logging import Telemetry
from model.waypoint import Waypoint
import math

import time
from scenario_builder import ScenarioBuilder, ScenarioActor

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

def downsample_waypoints(waypoints: List[Waypoint]) -> List[Waypoint]:
    res = []
    division = max(1, math.floor(len(waypoints) / 20))

    i = 0
    for p in waypoints:
        if i % division == 0:
            res.append(p)
            i += 1

    if len(waypoints) > 0:
        res.append(waypoints[len(waypoints) - 1])
    return res



def main(argc: int, argv: List[str]) -> int:
    
    coord = CoordinateConverter(COORD_ORIGIN)

    result = Telemetry.read_planning_result(1)
   
    ds_path = downsample_waypoints(result.path)
    ideal_motion_path = coord.convert_waypoint_path_to_map_pose(result.ego_location, ds_path)
    
    #tst.show_path(ideal_motion_path)
    
    current_pose = MapPose.from_str("-82.11653900146484|-2.0242035388946533|0.02852334827184677|-0.5942687392234802")
    pos = MapPose.find_nearest_goal_pose( current_pose , ideal_motion_path, 0)

    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))