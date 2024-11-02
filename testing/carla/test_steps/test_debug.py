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
import json

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


def _build_path( parent_x: int, parent_z: int, x: int, z: int) -> list[Waypoint]:
        dx = abs(x - parent_x)
        dz = abs(z - parent_z)
        num_steps = max(dx , dz)
        
        if num_steps == 0:
            return None
        
        dxs = dx / num_steps
        dzs = dz / num_steps
        
        last_x = parent_x
        last_z = parent_z
        
        path: list[Waypoint] = []
        
        if x < parent_x:
            dxs = -dxs
        if z < parent_z:
            dzs = -dzs

        for i in range(1, num_steps):            
            delta_x = dxs * i
            delta_z = dzs * i
            
            x = math.floor(parent_x + delta_x)
            z = math.floor(parent_z + delta_z)
            
            if x == last_x and z == last_z:
                continue
            
            path.append(
                Waypoint(x, z)
            )
        return path

def dump_result(og, path: list[Waypoint]):
    frame = og.get_color_frame()
    
    for p in path:
        frame[p.z, p.x, :] = [255, 255, 255]
    
    cv2.imwrite("debug_rrt.png", frame)

def _interpolate_path(og: OccupancyGrid, sparse_path: list[Waypoint]) -> list[Waypoint]:
    parent = sparse_path[0]
    res = []
        
    for i in range(1, len(sparse_path)):
        x, z = sparse_path[i].x, sparse_path[i].z
        path = _build_path(parent.x, parent.z, x, z)
        res.append(Waypoint(parent.x, parent.z))
        res.extend(path)
        res.append(Waypoint(x, z))
        parent = sparse_path[i]
        dump_result(og, res)
        
        
    #res.reverse()
    return res
    
def _post_process_smooth(og, sparse_path: list[Waypoint]):
        
    path = None
    if (len(sparse_path) >= 5):
        try:
            sparse_path.reverse()
            smooth_path = WaypointInterpolator.path_smooth_rebuild(sparse_path, 1)
            if og.check_all_path_feasible(smooth_path):
                return smooth_path
            else:
                sparse_path.reverse()
        except Exception as e:
            print(e)
            path = None
            sparse_path.reverse()
        
    path = _interpolate_path(og, sparse_path)
    if len(path) < 5:
        return None
       
    s_try = [30, 20, 10, 1]
            
    for s in s_try:
        try:
            smooth_path = WaypointInterpolator.path_smooth_rebuild(path, s)
            if len(smooth_path) == 0:
                continue
            dump_result(og, smooth_path)
            if og.check_all_path_feasible(smooth_path):
                return smooth_path
        except:
            continue
            
    if og.check_all_path_feasible(path):
        return path
    return None    

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
    
    path = [Waypoint.from_str(k) for k in json.loads('["(128, 107, 0)", "(115, 79, 0)", "(133, 55, 0)", "(106, 41, 0)", "(106, 35, 0)", "(107, 17, 0)", "(108, 6, 0)"]')]
   
    path = _post_process_smooth(og, path)
    
    dump_result(og, path)
    
    # original_heading, _ = compute_heading(p1.x, p1.z, p2.x, p2.z, PhysicalParameters.OG_WIDTH,
    #     PhysicalParameters.OG_HEIGHT)
    
    # print (f"original_heading = {math.degrees(original_heading)}")
    
    # for p in path:
    #     color_f[p.z, p.x, :] = [255, 255, 255]
    #     h, _ = compute_heading(p1.x, p1.z, p.x, p.z, PhysicalParameters.OG_WIDTH,
    #     PhysicalParameters.OG_HEIGHT)
    #     print (f"({p.x}, {p.z}) heading = {math.degrees(h)}")
        
    # cv2.imwrite("debug_dump.png", color_f)

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))