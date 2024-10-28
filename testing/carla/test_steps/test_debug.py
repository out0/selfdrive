import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.waypoint import Waypoint
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
import json

def main(argc: int, argv: List[str]) -> int:
    
    log = '["(128, 107, 0)", "(128, 107, 0)", "(128, 103, 0)", "(128, 100, 0)", "(128, 97, 0)", "(128, 93, 0)", "(128, 90, 0)", "(128, 87, 0)", "(129, 83, 0)", "(129, 80, 0)", "(129, 77, 0)", "(129, 74, 0)", "(129, 70, 0)", "(129, 67, 0)", "(130, 64, 0)", "(130, 60, 0)", "(130, 57, 0)", "(130, 54, 0)", "(130, 50, 0)", "(130, 47, 0)", "(131, 44, 0)", "(131, 41, 0)", "(131, 37, 0)", "(131, 34, 0)", "(131, 31, 0)", "(131, 27, 0)", "(132, 24, 0)", "(132, 21, 0)", "(132, 17, 0)", "(132, 14, 0)", "(132, 11, 0)", "(132, 8, 0)", "(133, 4, 0)"]'
    str_path = json.loads(log)
   
    path = []
    for p in str_path:
       path.append(Waypoint.from_str(p))
    
    smooth_path = WaypointInterpolator.path_smooth_rebuild(path)
    
    z = 1
    

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))