import numpy as np
import random, time
from test_utils import TestFrame, TestData, TestUtils
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import math
from node_grid import NodeGrid
from basic_types import *


Waypoint = tuple[int, int, float]

SEGMENTATION_COST = [-1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1]


    

class RRT:
    
    _width: int
    _height: int
    _perception_width_m: float
    _perception_height_m: float
    _max_steering_angle_deg: float
    _vehicle_length_m: float
    _timeout_ms: int
    _min_dist_x: int
    _min_dist_z: int
    _lower_bound_x: int
    _lower_bound_z: int
    _upper_bound_x: int
    _upper_bound_z: int
    _max_path_size_px: float
    _dist_to_goal_tolerance_px: float
    _start: tuple[int, int, float]
    _goal: tuple[int, int, float]
    _velocity_m_s: float
    _img: np.ndarray
    _nodes: NodeGrid
    _goal_reached: bool
    _proc_start: float
    _segmentation_class_cost: list[float]
    _check_min_distances: bool
    _check_kinematic_constraints: bool
    _goal_node_reached: int2

    
    def __init__(self, 
              width: int, 
              height: int,
              perception_width_m: float,
              perception_height_m: float,
              max_steering_angle_deg : float,
              vehicle_length_m: float,
              timeout_ms: int,
              min_dist_x: int,
              min_dist_z: int,
              lower_bound_x: int,
              lower_bound_z: int,
              upper_bound_x: int,
              upper_bound_z: int,
              max_path_size_px: float = 30.0,
              dist_to_goal_tolerance_px: float = 5.0
              ):
        self._width = width
        self._height = height
        self._perception_width_m = perception_width_m
        self._perception_height_m = perception_height_m
        self._max_steering_angle_deg = max_steering_angle_deg
        self._vehicle_length_m = vehicle_length_m
        self._timeout_ms = timeout_ms
        self._min_dist_x = int(min_dist_x/2)
        self._min_dist_z = int(min_dist_z/2)
        self._lower_bound_x = lower_bound_x
        self._lower_bound_z = lower_bound_z
        self._upper_bound_x = upper_bound_x
        self._upper_bound_z = upper_bound_z
        self._max_path_size_px = max_path_size_px
        self._dist_to_goal_tolerance_px = dist_to_goal_tolerance_px
        self._proc_start = 0
        self._check_min_distances = False
        self._check_kinematic_constraints = False
        self._segmentation_class_cost = None
        
        
    def set_plan_data(self, img: np.ndarray, start: Waypoint, goal: Waypoint, velocity_m_s: float) -> bool:
        self._img = img
        self._start = start
        self._goal = goal
        self._velocity_m_s = velocity_m_s
        
   
    def search_init(self) -> None:
        self._nodes = NodeGrid(self._width, self._height, (self._start[0], self._start[1]))
        self._nodes.set_search_params((self._min_dist_x, self._min_dist_z),
                                      (self._lower_bound_x, self._lower_bound_z),
                                      (self._upper_bound_x, self._upper_bound_z))
        
        if self._check_min_distances:
            self._nodes.set_class_costs(self._segmentation_class_cost)

        if self._check_kinematic_constraints:
            self._nodes.set_physical_params(PhysicalParameters(
                (self._width / self._perception_width_m, self._height / self._perception_height_m),
                self._vehicle_length_m/2,
                math.radians(self._max_steering_angle_deg)
        ))
        
        self._goal_reached = False
        self._proc_start = time.time()
        
    def enable_min_distances_check(self, segmentation_class_cost: list[float]) -> None:
        self._segmentation_class_cost = segmentation_class_cost
        self._check_min_distances = True

    def enable_kinematic_constraints_check(self) -> None:
        self._check_kinematic_constraints = True
        
    
    def __sample(self) -> int2:
        return (
            random.randint(0, self._width - 1),
            random.randint(0, self._height - 1)
        )
    
    
    def __dist(self, p1: int2, p2: int2) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.hypot(dx, dy)
    
    def __clip_sample(self, parent: int2, node: int2, max_step_size: float) -> int2:
        dx = node[0] - parent[0]
        dy = node[1] - parent[1]
        dist = np.hypot(dx, dy)
        if dist <= max_step_size:
            return (node[0], node[1])
        theta = np.arctan2(dy, dx)
        new_x = int(parent[0] + max_step_size * np.cos(theta))
        new_y = int(parent[1] + max_step_size * np.sin(theta))
        return (new_x, new_y)
    
    def __check_pos_limits(self, node: int2) -> bool:
        xc = node[0]
        zc = node[1]
        return xc >= 0 and xc < self._width and zc >= 0 and zc < self._height
    
    def __interpolate_straight_line(self, p1: int2, p2: int2) -> list[int2]:
        path = []
        
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        
        if dz == 0:
            return []
        
        steps = int(dz)
        
        for i in range(steps + 1):
            x = int(p1[0] + (dx * i) / steps)
            z = int(p1[1] + (dz * i) / steps)
            path.append((x, z))
        
        return path
    
    def __check_min_distances(self, path: list[int2]) -> bool:
        
        for p in path:
            if not self._nodes.check_feasible(self._img, p):
                return False
        return True
    
    def connect_new_node(self, parent: int2, node: int2, dist: float) -> bool:
        if not self.__check_pos_limits(node):
            return False
        
        if self._check_kinematic_constraints:
            res = self._nodes.check_graph_connection(self._img, parent, node, self._velocity_m_s)
            if res.feasible:
                self._nodes.add_node(node, parent, res.cost)
                self._nodes.set_heading(node[0], node[1], res.final_heading)
                return True
        else:
            if self._check_min_distances:
                sub_path = self.__interpolate_straight_line(parent, node)
                if not self.__check_min_distances(sub_path):
                    return False
                
                self._nodes.add_node(node, parent, dist)
        
        return True
    
    def __check_timeout(self) -> bool:
        if self._timeout_ms <= 0:
            return False
        elapsed_time = (time.time() - self._proc_start) * 1000
        return elapsed_time > self._timeout_ms
    
    def __sample_search_space(self) -> int2:
        new_node: int2 = self.__sample()
        nearest_parent_f, dist = self._nodes.find_nearest(new_node)
        nearest_parent = (int(nearest_parent_f[0]), int(nearest_parent_f[1]))
        new_node = self.__clip_sample(nearest_parent, new_node, self._max_path_size_px)
        
        if self.connect_new_node(parent=nearest_parent, node=new_node, dist=dist):
            return new_node
        
        return None
    
    def __check_goal_reached(self, new_node: int2) -> bool:
        dist_to_goal = self.__dist(new_node, self._goal)
        
        if dist_to_goal < self._dist_to_goal_tolerance_px:
            self._goal_node_reached = new_node
            self._goal_reached = True
            # to_goal_node = self.__clip_sample(new_node, self._goal, self._max_path_size_px)
            
            # if self.connect_new_node(new_node, to_goal_node):
            #     self._nodes.add_node(to_goal_node, new_node, dist_to_goal)
            #     self._goal_reached = True
    
    def loop(self) -> bool:
        
        if (self.__check_timeout()):
            return False

        new_node = self.__sample_search_space()
        if new_node is None:
            return True
        
        self.__check_goal_reached(new_node)

        return not self._goal_reached
       
    def loop_optimize(self) -> bool:
        pass
    
    def goal_reached(self) -> bool:
        return self._goal_reached
    
    def get_planned_path(self, interpolate: bool = False):
        nearest_parent, dist = self._nodes.find_nearest((self._goal[0], self._goal[1]))
        
        path = []
        node = (int(nearest_parent[0]), int(nearest_parent[1]))
        while node[0] != -1 and node[1] != -1:
            path.append(node)
            node = self._nodes.get_parent(node)
        
        
        if interpolate and len(path) > 1:
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            
            # Generate a parameter t for cubic spline interpolation
            t = np.arange(len(path))
            
            # Create cubic splines for x and y
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            
            # Interpolate with finer granularity
            t_fine = np.linspace(0, len(path) - 1, num=10 * len(path))
            x_fine = cs_x(t_fine)
            y_fine = cs_y(t_fine)
            
            # Combine interpolated points into a new path
            path = [(int(xf), int(yf)) for xf, yf in zip(x_fine, y_fine)]
        
        path.reverse()
        return path
    

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = -1


def dev_frame() -> np.ndarray:
    #return np.full((1024, 1024, 3), 1.0, dtype=np.float32)
    return np.full((256, 256, 3), 1.0, dtype=np.float32)
    
def test_regular_rrt():
    img = TestUtils.timed_exec(dev_frame)
        
        
    TestUtils.output_path_result(img, None, "output1.png")
    rrt = RRT(
        width=256,
        height=256,
        perception_height_m=OG_REAL_HEIGHT,
        perception_width_m=OG_REAL_WIDTH,
        max_steering_angle_deg=MAX_STEERING_ANGLE,
        vehicle_length_m=VEHICLE_LENGTH_M,
        timeout_ms=TIMEOUT,
        min_dist_x=22,
        min_dist_z=40,
        lower_bound_x=119,
        lower_bound_z=148,
        upper_bound_x=137,
        upper_bound_z=108,
        max_path_size_px=40.0,
        dist_to_goal_tolerance_px=15.0
        )
    
    rrt.set_plan_data(img, (128, 128, 0), (128, 0, 0), 1)
    #rrt.set_plan_data(img, (512, 512, 0), (128, 0, 0), 1)
    
    rrt.enable_min_distances_check(SEGMENTATION_COST)
    rrt.enable_kinematic_constraints_check()
    
    loop = True
    while loop:
        start_time = time.time()
        rrt.search_init()
        while (not rrt.goal_reached() and rrt.loop()):
            #time.sleep(0.5)
            pass
            
        end_time = time.time()

        #self.assertTrue(rrt.goal_reached())
        print(f"goal reached? {rrt.goal_reached()}")

        execution_time = end_time - start_time  # Calculate the time taken
        print(f"Coarse path: {1000*execution_time:.6f} ms")
        
        path = rrt.get_planned_path(True)
        
        TestUtils.output_path_result(img, path, "output1.png")

        #loop = False    
    
    
if __name__ == "__main__":
    test_regular_rrt()