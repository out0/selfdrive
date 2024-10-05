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
from utils.smoothness import Smoothness2D

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = -1
PLANNER_TYPE = LocalPlannerType.Overtaker

class Statistics:
    cost: float
    path_size: float
    exec_time: float
    
    def __init__(self) -> None:
        self.cost = 0
        self.path_size = 0
        self.exec_time = 0

def compute_statistics (seq: int) -> Statistics:
    coord = CoordinateConverter(COORD_ORIGIN)

    result = Telemetry.read_planning_result(seq)
    if result is None:
        return None    
    
    sm = Smoothness2D()    
    for p in result.path:
        sm.add_point(p.x, p.z)
        
    stats = Statistics()
    stats.cost = sm.get_cost()
    
    map_path = coord.convert_waypoint_path_to_map_pose(result.ego_location, result.path)
    last = map_path[0]
    stats.path_size = 0
    
    for i in range(1, len(map_path)):
        curr = map_path[i]
        stats.path_size += MapPose.distance_between(last, curr)
        last = curr
    
    stats.exec_time = result.total_exec_time_ms
    return stats



def main(argc: int, argv: List[str]) -> int:
    
    full_stats = Statistics()
    count = 0

    for i in range(1,1000):
        stats = compute_statistics(i)
        if stats is None:
            break
        full_stats.cost += stats.cost
        full_stats.exec_time += stats.exec_time
        full_stats.path_size += stats.path_size
        count += 1
    
    if count == 0:
        print ("no data found")
        return
        
    full_stats.cost = full_stats.cost / count
    full_stats.exec_time = full_stats.exec_time / count

    print(f"avg cost: {full_stats.cost:.2f}")
    print(f"avg time: {full_stats.exec_time:.2f} ms")
    print(f"path_size: {full_stats.path_size:.2f} m")
    
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))