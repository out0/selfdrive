import sys
sys.path.append("../../../")
from model.planning_data import PlanningData
import cv2, numpy as np, json
from vision.occupancy_grid_cuda import SEGMENTED_COLORS
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose

class PlannerDataReader:

    @classmethod
    def __convert_frame(cls, colored_f: np.ndarray) -> np.ndarray:
        original_f = np.zeros(colored_f.shape, dtype=np.uint8)
        for i in range (0, colored_f.shape[0]):
            for j in range (0, colored_f.shape[1]):
                for k in range (0, len(SEGMENTED_COLORS)):
                    if colored_f[i, j, 0] == SEGMENTED_COLORS[k][0] and\
                        colored_f[i, j, 1] == SEGMENTED_COLORS[k][1] and\
                        colored_f[i, j, 2] == SEGMENTED_COLORS[k][2]:
                        original_f[i, j, 0] = k
                        break
        return original_f      

    @classmethod
    def read (cls, log_num: int) -> tuple[WorldPose, PlanningData]:
        bev = cls.__convert_frame(cv2.imread(f"imgs/bev_{log_num}.png"))
        with open(f"imgs/log_{log_num}.log") as f:
            
            lines = f.readlines()
            log_data = json.loads(lines[0])
            pos_log_data = json.loads(lines[1])
            
            world_pose = WorldPose.from_str(log_data["gps"])
            
            return world_pose, PlanningData(
                bev,
                ego_location=MapPose.from_str(log_data["pose"]),
                velocity=float(log_data["velocity"]),
                goal=MapPose.from_str(pos_log_data["goal"]),
                next_goal=MapPose.from_str(pos_log_data["next_goal"])
            )
            

class PlannerTestOutput:
    _frame: np.ndarray

    def __init__(self, frame: np.ndarray, convert_to_gray: bool = True):
        self._frame = frame
        
        if convert_to_gray:
            self._frame[:,:,1] = self._frame[:,:,0]
            self._frame[:,:,2] = self._frame[:,:,0]

    
    def add_point (self, point: Waypoint,  color = [255, 255, 255]) -> None:
        
        if point.x < 0:
            point.x = 0
        if point.z < 0:
            point.z = 0
        
        if point.x > 0:
            self._frame[point.z, point.x - 1, :] = color
        if point.x < self._frame.shape[1] - 1:
            self._frame[point.z, point.x + 1, :] = color
        if point.z > 0:
            self._frame[point.z - 1, point.x, :] = color
        if point.z < self._frame.shape[0] - 1:
            self._frame[point.z + 1, point.x, :] = color

        self._frame[point.z, point.x, :] = color
    

    def add_path (self, path: list[Waypoint], color = [255, 255, 255]) -> None:
        for p in path:
            self.add_point(p, color)
       
    def write (self, file: str) -> None:
        cv2.imwrite(file, self._frame)