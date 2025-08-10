from pydriveless import MapPose
from ensemble import PlanningResult, PlanningData
import cv2

def read_path (file: str) -> list[MapPose]:
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    path = []
    for l in lines:
        p = l.replace("\n", "").split(';')
        path.append(MapPose(x=float(p[0]), y=float(p[1]), z=float(p[2]), heading=float(p[3])))
    return path


def export_planning_response(file: str, planner_data: PlanningData, res: PlanningResult):
    frame = planner_data.og().get_color_frame()
    if res is not None:
        for p in res.path:
            frame[int(p.z), int(p.x)] = [255, 255, 255]
    
    goal = planner_data.local_goal()

    # Draw a cross at the local goal position (goal.x, goal.z)
    cross_size = 2  # Length of cross arms in pixels
    color = [0, 255, 0]  # Green

    for dx in range(-cross_size, cross_size + 1):
        x = int(goal.x) + dx
        z = int(goal.z)
        if 0 <= x < frame.shape[1] and 0 <= z < frame.shape[0]:
            frame[z, x] = color  # Horizontal line

    for dz in range(-cross_size, cross_size + 1):
        x = int(goal.x)
        z = int(goal.z) + dz
        if 0 <= x < frame.shape[1] and 0 <= z < frame.shape[0]:
            frame[z, x] = color  # Vertical line

    cv2.imwrite(file, frame)