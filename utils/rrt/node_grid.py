import numpy as np
import random, time
from test_utils import TestFrame, TestData, TestUtils
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import math
from basic_types import *


def convert_map_pose_to_waypoint(center: int2, rate: float2, coord: float2) -> int2:
    return (
        center[0] + int(coord[1] * rate[0]),
        center[1] - int(coord[0] * rate[1]))
    
def convert_waypoint_to_map_pose(center: int2, inv_rate: float2, coord: int2) -> float2:
    return (
        inv_rate[1] * (center[1] - coord[1]),
        inv_rate[0] * (coord[0] - center[0])
    )

def euclidean_distance(start: int2, end: int2) -> float:
    return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

def compute_path_heading(start: float2, end: float2) -> float:
    dy = end[1] - start[1]
    dx = end[0] - start[0]

    if dy >= 0 and dx > 0:  # Q1
        return math.atan(dy / dx)
    elif dy >= 0 and dx < 0:  # Q2
        return math.pi - math.atan(dy / abs(dx))
    elif dy < 0 and dx > 0:  # Q3
        return -math.atan(abs(dy) / dx)
    elif dy < 0 and dx < 0:  # Q4
        return math.atan(dy / dx) - math.pi
    elif dx == 0 and dy > 0:
        return math.pi / 2
    elif dx == 0 and dy < 0:
        return -math.pi / 2
    return 0.0

def clip(val: float, min: float, max: float) -> float:
    if (val < min):
        return min
    if (val > max):
        return max
    return val


class GraphConnectionResult:
    final_heading: float
    feasible: bool
    cost: float
    
    def __init__(self, final_heading: float, feasible: bool, cost: float):
        self.final_heading = final_heading
        self.feasible = feasible
        self.cost = cost


class NodeGrid:
    _grid: np.ndarray
    _grid_data: np.ndarray
    _search_tree: KDTree
    _node_list: list[int2]
    _width: int
    _height: int
    _class_costs: list[float]
    _min_dist: int2
    _lower_bound: int2
    _upper_bound: int2
    _physical_params: PhysicalParameters
    
    def __init__(self, width: int, height: int, start: int2):
        self._grid = np.zeros((height, width, 3), dtype=np.int32)
        self._grid_data = np.zeros((height, width, 2), dtype=np.float32)
        self._node_list = None
        self.add_node(start, (-1, -1), 0.0)
        self._width = width
        self._height = height
        self._class_costs = None
        self._physical_params = None
        self._center = (int(width/2), int(height/2))
        
    
    def add_node(self, node: int2, parent: int2, cost: float) -> None:
        x = node[0]
        z = node[1]
        if self._grid[z, x, 2] == 1:
            return
        self._grid[z, x, 0] = parent[0]
        self._grid[z, x, 1] = parent[1]
        self._grid[z, x, 2] = 1
        self._grid_data[z, x, 0] = cost
        
        if (self._node_list is None):
            self._node_list = [node]
        else:
            self._node_list.append(node)
        self._search_tree = KDTree(self._node_list)
        
    def set_cost(self, x: int, z: int, cost: float) -> None:
        self._grid_data[z, x, 0] = cost
    
    def get_cost(self, x: int, z: int) -> float:
        return self._grid_data[z, x, 0]
    
    def __check_pos_limits(self, node: int2) -> bool:
        xc = node[0]
        zc = node[1]
        return xc >= 0 and xc < self._width and zc >= 0 and zc < self._height
    
    def set_heading(self, x: int, z: int, heading_rad: float) -> None:
        self._grid_data[z, x, 1] = heading_rad
    
    def get_heading(self, x: int, z: int) -> float:
        return self._grid_data[z, x, 1]
        
    def find_nearest(self, node: int2) -> tuple[int2, float]:
        dist, idx = self._search_tree.query(node)
        nearest_node = self._search_tree.data[idx]
        return (nearest_node[0], nearest_node[1]), dist

    def get_parent(self, node: int2) -> int2:
        x = node[0]
        z = node[1]
        parent_x = self._grid[z, x, 0]
        parent_z = self._grid[z, x, 1]
        return (int(parent_x), int(parent_z))
    
    def __in_ego_boundaries(self, x: int, z: int) -> bool:
        return (
            self._lower_bound[0] <= x <= self._upper_bound[0] and
            self._upper_bound[1] <= z <= self._lower_bound[1])
    
    def check_feasible(self, frame: np.ndarray, p: int2) -> bool:
        for z in range(p[1] - self._min_dist[1], p[1] + self._min_dist[1] + 1):
            for x in range(p[0] - self._min_dist[0], p[0] + self._min_dist[0] + 1):
                if not self.__check_pos_limits((x, z)):
                    continue
                
                if self.__in_ego_boundaries(x, z):
                    continue
                
                segmentation_class = int(frame[z, x, 0])
                if self._class_costs[segmentation_class] < 0:
                    return False
        return True
    
    def set_class_costs(self, class_costs: list[float]) -> None:
        self._class_costs = class_costs
        
    def set_search_params(self, min_dist: int2, lowerbound: int2, upperbound: int2) -> None:
        self._min_dist = min_dist
        self._lower_bound = lowerbound
        self._upper_bound = upperbound
        
        
    def set_physical_params(self, physical_params: PhysicalParameters) -> None:
        self._physical_params = physical_params
        
    
    def check_graph_connection(
        self,
        frame: np.ndarray,
        start: int2,
        end: int2,
        velocity: float) -> GraphConnectionResult:
        
        WIDTH = frame.shape[1]
        HEIGHT = frame.shape[0]
        
        START_MAP: float2 = convert_waypoint_to_map_pose(self._center, self._physical_params.inv_rate, start)
        END_MAP: float2 = convert_waypoint_to_map_pose(self._center, self._physical_params.inv_rate, end)
        DT = 0.1        
        DS = velocity * DT
        DISTANCE = euclidean_distance(start, end)
        TOTAL_STEPS = int(round(DISTANCE / DS))
        
        heading = self.get_heading(start[0], start[1])
        path_heading = compute_path_heading(START_MAP, END_MAP)
        steering_angle_rad = clip(path_heading - heading, -self._physical_params.maxSteering_rad, self._physical_params.maxSteering_rad)
        
        nextp_m: float2 = (START_MAP[0], START_MAP[1])
        nextp: int2 = (-1, -1)
        lastp: int2 = (start[0], start[1])
        best_end_dist = DISTANCE
        path_cost = 0
        curr_cost = self.get_cost(start[0], start[1])
        
        for _ in range(TOTAL_STEPS):
            steer = math.tan(steering_angle_rad)
            beta = math.atan(steer / self._physical_params.lr)
            nextp_m = (
                nextp_m[0] + DS * math.cos(heading + beta),
                nextp_m[1] + DS * math.sin(heading + beta))
            heading += DS * math.cos(beta) * steer / (2 * self._physical_params.lr)
            
            path_heading = compute_path_heading(nextp_m, END_MAP)
            steering_angle_rad = clip(path_heading - heading, -self._physical_params.maxSteering_rad, self._physical_params.maxSteering_rad)
            dist = euclidean_distance(nextp_m, END_MAP)
            nextp = convert_map_pose_to_waypoint(self._center, self._physical_params.rate, nextp_m)
            
            if (nextp[0] == lastp[0] and nextp[1] == lastp[1]):
                continue

            if (nextp[0] < 0 or nextp[0] >= WIDTH):
                return GraphConnectionResult(heading, False, path_cost)
            if (nextp[1] < 0 or nextp[1] >= HEIGHT):
                return GraphConnectionResult(heading, False, path_cost)
            
            if (best_end_dist < dist):
                path_cost = curr_cost
                return GraphConnectionResult(heading, best_end_dist <= 2, path_cost)
            
            if not self.check_feasible(frame, nextp):
                return GraphConnectionResult(heading, False, path_cost)
            
            lastp = (nextp[0], nextp[1])
            best_end_dist = dist
                
        return GraphConnectionResult(heading, False, path_cost)