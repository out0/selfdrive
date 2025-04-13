import sys
sys.path.append("../../../")
sys.path.append("../../../../")
import cv2, numpy as np, json
import math
from utils.cudac.cuda_frame import CudaFrame
from model.physical_parameters import PhysicalParameters
from model.waypoint import Waypoint
from utils.fast_rrt.fastrrt import FastRRT
import time

PROPORTION_meters_per_px = 0.135316469
GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3

class TestData:
    frame: CudaFrame
    start: Waypoint
    goal: Waypoint
    upper_bound: Waypoint
    lower_bound: Waypoint
    
    def __init__(self, frame: CudaFrame, start: Waypoint, goal: Waypoint, upper_bound: Waypoint = None, lower_bound: Waypoint = None):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.frame = frame
        self.start = start
        self.goal = goal
    
    def width(self) -> int:
        return self.frame.get_shape()[1]

    def height(self) -> int:
        return self.frame.get_shape()[0]
    
    def real_width(self) -> float:
        return self.width() * PROPORTION_meters_per_px
    
    def real_height(self) -> float:
        return self.height() * PROPORTION_meters_per_px


class TestFrame:
    __file: str
    __vehicle_dimensions: tuple[int, int]
    
    def __init__(self, file: str, vehicle_dimensions: tuple[int, int] = (18, 40)):
        self.__file = file
        self.__vehicle_dimensions = vehicle_dimensions
        
    def get_test_data(self):
        return self.__process_frame()
    
    def __is_empty(self, p):
        return p[0] == 255 and p[1] == 255 and p[2] == 255
    
    def __is_start(self, p):
        return p[0] == 0 and p[1] == 255 and p[2] == 0
        
    def __is_goal(self, p):
        return p[0] == 0 and p[1] == 0 and p[2] == 255
        
    def __process_frame(self):
        frame = np.array(cv2.imread(self.__file))
        if frame is None:
            raise ValueError(f"Could not read image from {self.__file}")
        
        new_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        start = None
        goal = None
        
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if (i == 428 and j == 265):
                    pass
                if self.__is_empty(frame[i, j]):
                    new_frame[i, j] = [1.0, 0, 0]
                elif self.__is_start(frame[i, j]):
                    start = Waypoint(x=j, z=i, heading=0.0)
                    new_frame[i, j] = [1.0, 0, 0]
                elif self.__is_goal(frame[i, j]):
                    goal = Waypoint(x=j, z=i, heading=0.0)
                    new_frame[i, j] = [1.0, 0, 0]
                else:
                    new_frame[i, j] = [0, 0, 0]
                    
        
        lower_bound = Waypoint(x=start.x - int(self.__vehicle_dimensions[0]/2), z=start.z + int(self.__vehicle_dimensions[1]/2))        
        upper_bound = Waypoint(x=start.x + int(self.__vehicle_dimensions[0]/2), z=start.z - int(self.__vehicle_dimensions[1]/2))
        
        
        cuda_f = CudaFrame(frame = new_frame, 
                        min_dist_x= PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                        min_dist_z= PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                        lower_bound= lower_bound,
                        upper_bound= upper_bound)
        
        return TestData(frame=cuda_f, start=start, goal=goal, upper_bound=upper_bound, lower_bound=lower_bound)
        

class TestUtils:
    def log_graph(rrt: FastRRT, frame: CudaFrame, file: str) -> None:
        cimg = frame.get_color_frame()
        nodes = rrt.export_graph_nodes()
            
        width = cimg.shape[1]
        height = cimg.shape[0]
            
        for n in nodes:
            x = n[0]
            z = n[1]
            if (x < 0 or x >= width): continue
            if (z < 0 or z >= height): continue
            
            if int(n[2]) == GRAPH_TYPE_NODE:
                cimg[z, x, :] = [255, 255, 255]
            elif int(n[2]) == GRAPH_TYPE_PROCESSING:
                cimg[z, x, :] = [0, 255, 0]
            elif int(n[2]) == GRAPH_TYPE_TEMP:
                cimg[z, x, :] = [0, 0, 255]     
            
        cv2.imwrite(file, cimg)
    
    def output_path_result(frame: CudaFrame, path: np.ndarray, output: str) -> None:
        
        f = frame.get_color_frame()
        
        for i in range(path.shape[0]):
            x = int(path[i,0])
            z = int(path[i,1])
            f[z, x, :] = [255, 255 , 255]
        
        cv2.imwrite(output, f)
        
    def output_obstacle_graph(frame: CudaFrame, output: str) -> None:
        shape = frame.get_shape()
        frame.invalidate_cpu_frame()
        orig_frame = frame.get_frame()
        raw = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (orig_frame[i, j, 2] == 1.0):
                    raw[i, j] = [255, 255, 255]
                else:
                    raw[i, j] = [0, 0, 0]
        cv2.imwrite(output, raw)
        
    def timed_exec(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{func.__name__}] {1000 * execution_time:.6f} ms")
        return result


if __name__ == "__main__":
    test_frame = TestFrame(file = "custom1.png")
    p = test_frame.get_test_data()
    j = 1    
  