import sys
sys.path.append("../../../")
from model.planning_data import PlanningData
import cv2, numpy as np, json
from vision.occupancy_grid_cuda import SEGMENTED_COLORS
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
import math

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
    _width: int
    _height: int

    def __init__(self, frame: np.ndarray, convert_to_gray: bool = True):
        self._frame = frame
        
        if convert_to_gray:
            self._frame[:,:,1] = self._frame[:,:,0]
            self._frame[:,:,2] = self._frame[:,:,0]
        
        self._width = self._frame.shape[1]
        self._height = self._frame.shape[0]

    
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
    
    def draw_vector(self, p1: Waypoint, heading: float, size = 10):
        r = math.radians(heading - 90)
        c = math.cos(r)
        s = math.sin(r)
        for i in range (size):
            xl = round(p1.x + c * i)
            zl = round(p1.z + s * i)
            if xl < 0 or xl >= self._frame.shape[1]: continue
            if zl < 0 or zl >= self._frame.shape[0]: continue
            self._frame[zl, xl, :] = [255, 0, 0]
    
    # def draw_arrow(self, position: Waypoint, angle_rad: float, length=20, color=(255, 0, 0)):

    #     # Ensure the arrow starts within the image bounds
    #     if position.x < 0 or position.x >= self._width or position.z < 0 or position.z >= self._height:
    #         raise ValueError("Position is out of image bounds.")

    #     # Convert angle to radians
    #     # Calculate the end point of the arrow
    #     end_x = int(position.x + length * np.cos(angle_rad))
    #     end_z = int(position.z + length * np.sin(angle_rad))  # y is inverted in image coordinates

    #     # Clip the end point to be within the image bounds
    #     end_x = np.clip(end_x, 0, self._width - 1)
    #     end_z = np.clip(end_z, 0, self._height - 1)

    #     # Draw the main arrow line
    #     PlannerTestOutput.draw_line(self._frame, (position.x, position.z), (end_x, end_z), color)

    #     # # Draw the arrowhead
    #     # angle = math.degrees(angle_rad)
    #     # arrow_head_length = 5
    #     # arrow_head_angle = 30  # degrees
    #     # for direction in [-1, 1]:
    #     #     head_angle = angle + direction * arrow_head_angle
    #     #     head_end_x = int(end_x + arrow_head_length * np.cos(np.deg2rad(head_angle)))
    #     #     head_end_y = int(end_z - arrow_head_length * np.sin(np.deg2rad(head_angle)))
    #     #     PlannerTestOutput.draw_line(self._frame, (end_x, end_z), (head_end_x, head_end_y), color)

    # def draw_line(image, start, end, color):
    #     x0, y0 = start
    #     x1, y1 = end
    #     dx = abs(x1 - x0)
    #     dy = abs(y1 - y0)
    #     sx = 1 if x0 < x1 else -1
    #     sy = 1 if y0 < y1 else -1
    #     err = dx - dy

    #     while True:
    #         # Ensure the pixel color is set correctly for RGB
    #         image[y0, x0] = color
    #         if (x0 == x1) and (y0 == y1):
    #             break
    #         err2 = err * 2
    #         if err2 > -dy:
    #             err -= dy
    #             x0 += sx
    #         if err2 < dx:
    #             err += dx
    #             y0 += sy

    

class TestFrame:
    frame: np.ndarray
    
    FREE_CLASS_TYPE = 1
    OBSTACLE_CLASS_TYPE = 0
    
    def __init__(self, width: int, height: int) -> None:
        self.frame = np.full((width, height, 3), TestFrame.FREE_CLASS_TYPE, dtype=np.int32)
    
    def dump_to_file (self, file: str = 'frame.png'):
        cv2.imwrite(file, self.frame)
        
    def read_from_file (self, file: str = 'frame.png'):
        self.frame = np.ndarray(cv2.imread(file, self._frame), dtype=np.int32)
        
    def add_obstacle (self, x1: int, z1: int, x2: int, z2: int) -> None:
        
        z_start = z1
        z_end = z2       
        if z1 > z2:
            z_start = z2
            z_end = z1

        x_start = x1
        x_end = x2       
        if x1 > x2:
            x_start = x2
            x_end = x1

        for z in range (z_start, z_end):
            for x in range (x_start, x_end):
                if z < 0 or x < 0: 
                    continue
                if z >= self.frame.shape[0] or x >= self.frame.shape[1]: 
                    continue
                self.frame[z, x, :] = [TestFrame.OBSTACLE_CLASS_TYPE, TestFrame.OBSTACLE_CLASS_TYPE, TestFrame.OBSTACLE_CLASS_TYPE]
                
    def add_squared_obstacle(self, x_center: int, z_center: int, size: int) -> None:
        l = size / 2
        self.add_obstacle(x_center - l, z_center - l, x_center + l, z_center + l)
    
    def add_point(self, x: int, z: int, color = [0, 0, 255]) -> None:
        for j in range (z - 1, z + 2):
            if j < 0: continue
            if j > self.frame.shape[0]: continue
            self.frame[j, x, :] = color
        
        for j in range (x - 1, x + 2):
            if x < 0: continue
            if j > self.frame.shape[0]: continue
            self.frame[z, j, :] = color            