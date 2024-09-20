import cv2
from model.planning_data import PlanningResult
from vision.occupancy_grid_cuda import OccupancyGrid
import numpy as np

def dump_result(og: OccupancyGrid, result: PlanningResult, color = [255, 255, 255]) -> None:
    if (result.path is None or len(result.path) == 0): 
        return
    
    frame = og.get_color_frame()

    for p in result.path:
        if p.x < 0 or p.x >= frame.shape[1]:
            continue
        if p.z < 0 or p.z >= frame.shape[0]:
            continue
            
        if p.x > 0:
            frame[p.z, p.x - 1, :] = color
        if p.x < frame.shape[1] - 1:
            frame[p.z, p.x + 1, :] = color
        if p.z > 0:
            frame[p.z - 1, p.x, :] = color
        if p.z < frame.shape[0] - 1:
            frame[p.z + 1, p.x, :] = color
    
        frame[p.z, p.x, :] = color
                        
    name = str.lower(result.planner_name).replace("*","").replace(" ","")
                        
    cv2.imwrite(f"plan_result_{name}.png", frame)
    dump_path(name, result)
    
def dump_path(name:str, result: PlanningResult) -> None:
    file = f"plan_result_{name}.log"
    with open(file, "w") as f:
        for p in result.path:
            f.write(f"{str(p)}\n")