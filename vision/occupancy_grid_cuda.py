from model.waypoint import Waypoint
import numpy as np
from utils.cudac.cuda_frame import *
from threading import Lock

class OccupancyGrid:
    _frame: CudaFrame
    _goal_point: Waypoint
    _minimal_distance_x: int
    _minimal_distance_z: int
    _frame_size: int
    _lock: Lock


    def __init__(self, frame: np.ndarray, minimal_distance_x: int, minimal_distance_z: int, lower_bound: Waypoint, upper_bound: Waypoint) -> None:
        self._frame = CudaFrame(frame, round(minimal_distance_x / 2), round(minimal_distance_z / 2), lower_bound, upper_bound)
        self._goal_point = None
        self._minimal_distance_x = minimal_distance_x / 2
        self._minimal_distance_z = minimal_distance_z / 2
        self._frame_size = frame.shape[0] * frame.shape[1] * frame.shape[2]
        self._lock = Lock()

    def clone(self) -> 'OccupancyGrid':
        new_frame = np.zeros(self._frame.shape)
        for i in range (0, self._frame.shape[0]):
            for j in range (0, self._frame.shape[1]):
                new_frame[i, j, 0] = self._frame[i, j, 0]
        return OccupancyGrid(new_frame)

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
        return int(self.get_frame()[z, x, 2]) & int(direction.value) > 0
    
    def check_direction_allowed(self, x: int, z: int, direction: int) -> bool:
        if self._frame is None:
            return False
        
        return int(self._frame.get_frame()[z, x, 2]) & direction > 0   

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