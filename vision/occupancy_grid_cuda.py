from model.waypoint import Waypoint
import numpy as np
from utils.cudac.cuda_frame import *
from threading import Lock


SEGMENTED_COLORS = np.array([
    [0,   0,   0],
    [128,  64, 128],
    [244,  35, 232],
    [70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [0,   0, 142],
    [0,   0,  70],
    [0,  60, 100],
    [0,  80, 100],
    [0,   0, 230],
    [119,  11,  32],
    [110, 190, 160],
    [170, 120,  50],
    [55,  90,  80],     # other
    [45,  60, 150],
    [157, 234,  50],
    [81,   0,  81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180]
])

class OccupancyGrid:
    _frame: CudaFrame
    _goal_point: Waypoint
    _minimal_distance_x: int
    _minimal_distance_z: int
    _lower_bound: Waypoint
    _upper_bound: Waypoint
    _frame_size: int
    _lock: Lock


    def __init__(self, frame: np.ndarray, minimal_distance_x: int, minimal_distance_z: int, lower_bound: Waypoint, upper_bound: Waypoint) -> None:
        self._frame = CudaFrame(frame, minimal_distance_x, minimal_distance_z, lower_bound, upper_bound)
        self._goal_point = None
        self._minimal_distance_x = minimal_distance_x
        self._minimal_distance_z = minimal_distance_z
        self._frame_size = frame.shape[0] * frame.shape[1] * frame.shape[2]
        self._lock = Lock()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def clone(self) -> 'OccupancyGrid':
        shape = self._frame.get_shape()
        new_frame = np.zeros(shape)
        frame = self._frame.get_frame()
        for i in range (0, shape[0]):
            for j in range (0, shape[1]):
                new_frame[i, j, 0] = frame[i, j, 0]
                
        return OccupancyGrid(
            frame=new_frame,
            minimal_distance_x=self._minimal_distance_x,
            minimal_distance_z=self._minimal_distance_z,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound
        )

    def set_goal(self, goal: Waypoint) -> None:
        self._lock.acquire()
        if  self._goal_point is not None and \
            self._goal_point.x == goal.x and \
            self._goal_point.z == goal.z:
                print (">> set_goal() - same goal, ignored")
                self._lock.release()
                return

        self._goal_point = goal
        self._frame.set_goal(goal)
        self._lock.release()
        

    def set_goal_vectorized(self, goal: Waypoint) -> None:
        self._lock.acquire()
        if  self._goal_point is not None and \
            self._goal_point.x == goal.x and \
            self._goal_point.z == goal.z:
                print (">> set_goal() - same goal, ignored")
                self._lock.release()
                return
        
        self._goal_point = goal
        self._frame.set_goal_vectorized(goal)
        self._lock.release()
        
    
    def check_direction_allowed(self, x: int, z: int, direction: GridDirection) -> bool:
        if self._frame is None:
            return False
        
        if  isinstance(direction, GridDirection):
            direction = int(direction.value)
        
        return int(self.get_frame()[z, x, 2]) & direction > 0
    
    
    def check_any_direction_allowed(self, x: int, z: int) -> bool:
        if self._frame is None:
            return False
               
        return int(self.get_frame()[z, x, 2]) != 0

    def get_color_frame(self) -> np.ndarray:
        self._lock.acquire()
        r = self._frame.get_color_frame()
        self._lock.release()
        return r

    def get_frame(self) -> np.ndarray:
        return self._frame.get_frame()
        
    def get_shape(self):
        if self._frame is None:
            return None
        
        return self._frame.get_shape()
    
    def width(self):
        if self._frame is None:
            return 0
        return self._frame.get_shape()[1]
    
    def height(self):
        if self._frame is None:
            return 0
        return self._frame.get_shape()[0]
    
    def get_goal_point(self) -> Waypoint:
        return self._goal_point 
    
    def get_minimal_distance_x(self) -> int:
        return  self._minimal_distance_x
   
    def get_minimal_distance_z(self) -> int:
        return  self._minimal_distance_z
  
    def check_path_feasible (self, path: list[Waypoint]) -> np.ndarray:
        return self._frame.compute_feasible_path(path)
    
    def check_all_path_feasible (self, path: list[Waypoint]) -> bool:
        r = self._frame.compute_feasible_path(path)
        for p in r:
            if not p:
                return False
        return True
    
    def check_waypoint_feasible(self, point: Waypoint) -> bool:
        return self._frame.check_waypoint_feasible(point)

    def find_best_cost_waypoint_with_heading(self, goal_x: int, goal_z: int, heading: float) -> Waypoint:
        return self._frame.find_best_cost_waypoint_with_heading(goal_x, goal_z, heading)
    
    def find_best_cost_waypoint(self, goal_x: int, goal_z: int) -> Waypoint:
        return self._frame.find_best_cost_waypoint(goal_x, goal_z)


    def find_car_dimensions(frame: np.ndarray, car_class: int) -> tuple[Waypoint, Waypoint]:
        mid_x = math.floor(frame.shape[1] / 2)
        mid_z = math.floor(frame.shape[0] / 2)
        
        i = mid_x
        while i >= 0 and frame[mid_z, i, 0] == car_class:
            i -= 1
        
        j = mid_z
        while j >= 0 and frame[j, mid_x, 0] == car_class:
            j -= 1
            
        x_size = mid_x - i
        z_size = mid_z - j
        
        bottom_left = Waypoint(mid_x - x_size, mid_z + z_size)
        top_right = Waypoint(mid_x + x_size, mid_z - z_size)
        
        return (bottom_left, top_right)
    
    def compute_heading(p1: 'Waypoint', p2: 'Waypoint') -> float:
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.degrees(math.pi/2 - math.atan2(-dz, dx))