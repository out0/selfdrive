import sys
sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.waypoint import Waypoint
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from model.physical_parameters import PhysicalParameters
import math
from vision.occupancy_grid_cuda import OccupancyGrid
import cv2

def compute_heading(p1_x, p1_y, p2_x, p2_y, width, height):
    valid = False

    # Check if points are the same or out of bounds
    if (p1_x == p2_x and p1_y == p2_y) or \
       (p1_x < 0 or p1_y < 0 or p2_x < 0 or p2_y < 0) or \
       (p1_x >= width or p1_y >= height or p2_x >= width or p2_y >= height):
        return 0.0, valid

    # Calculate heading
    dx = p2_x - p1_x
    dz = p2_y - p1_y
    valid = True
    heading = math.pi / 2 - math.atan2(-dz, dx)

    # Adjust heading if it's greater than 180 degrees
    if heading > math.pi:
        heading -= 2 * math.pi

    return heading, valid


def main(argc: int, argv: List[str]) -> int:
    
    p1 = Waypoint(128, 107)
    p2 = Waypoint(106, 86)
    
    path: list[Waypoint] = WaypointInterpolator.interpolate_straight_line_path2(
        p1,
        p2,
        PhysicalParameters.OG_WIDTH,
        PhysicalParameters.OG_HEIGHT,
        20
    )
    
    bev = cv2.imread("planning_data/bev_1.png")
    
    og = OccupancyGrid(
       bev,
       PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
       PhysicalParameters.MIN_DISTANCE_HEIGHT_PX, 
       PhysicalParameters.EGO_LOWER_BOUND,
       PhysicalParameters.EGO_UPPER_BOUND
    )
    
    color_f = og.get_color_frame()
    
    original_heading, _ = compute_heading(p1.x, p1.z, p2.x, p2.z, PhysicalParameters.OG_WIDTH,
        PhysicalParameters.OG_HEIGHT)
    
    print (f"original_heading = {math.degrees(original_heading)}")
    
    for p in path:
        color_f[p.z, p.x, :] = [255, 255, 255]
        h, _ = compute_heading(p1.x, p1.z, p.x, p.z, PhysicalParameters.OG_WIDTH,
        PhysicalParameters.OG_HEIGHT)
        print (f"({p.x}, {p.z}) heading = {math.degrees(h)}")
        
    cv2.imwrite("debug_dump.png", color_f)

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))