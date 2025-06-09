import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from planner.local_planner.local_planner import LocalPlannerType, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from utils.telemetry import Telemetry
from utils.jerk import CurveAssessment, CurveData
from model.physical_parameters import PhysicalParameters

VALID = True
NOT_VALID = False


COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = -1
#PLANNER_TYPE = LocalPlannerType.Overtaker

class Statistics:
    cost: float
    path_size: float
    exec_time: float
    jerk: float
    
    def __init__(self) -> None:
        self.cost = 0
        self.path_size = 0
        self.exec_time = 0
        self.jerk = 0

def compute_statistics (seq: int, scenario: int, planner: str) -> CurveData:
    #coord = CoordinateConverter(COORD_ORIGIN)


    result = Telemetry.read_planning_result(seq, base_path=f"/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/testing/carla/results/scen{scenario}/{planner}")
    if result is None:
        return None, VALID
    
    if result.result_type == PlannerResultType.TOO_CLOSE:
        return None, NOT_VALID
    
    data: CurveData = CurveAssessment.assess_curve(
        curve=result.path,
        start_heading=result.local_start.heading,
    )
    
    data.proc_time_ms = result.total_exec_time_ms
    
    return data, VALID
    

def main(argc: int, argv: List[str]) -> int:
    
    full_stats = Statistics()
    count = 0
    
    planner = "h-ensemble"

    for s in range(1, 7):
        for i in range(1,1000):
            stats, t =  compute_statistics(i, s, planner)
            
            if t == NOT_VALID:
                continue
            
            if stats is None:
                break
            
            full_stats.cost += 0
            full_stats.exec_time += stats.proc_time_ms
            full_stats.path_size += stats.total_length
            full_stats.jerk += stats.jerk
            count += 1
        
        if count == 0:
            print ("no data found")
            return
            
        full_stats.cost = full_stats.cost / count
        #full_stats.jerk = full_stats.jerk / count
        full_stats.exec_time = full_stats.exec_time / count

        #print(f"avg cost: {full_stats.cost:.2f}")
        print(f"scenario: {s}")
        print(f"total jerk: {full_stats.jerk:.2f} m/s³")
        print(f"avg time: {full_stats.exec_time:.2f} ms")
        print(f"path_size: {full_stats.path_size * PhysicalParameters.OG_HEIGHT_PX_TO_METERS_RATE:.2f} m")
        print("\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))