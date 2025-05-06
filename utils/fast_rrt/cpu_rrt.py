import numpy as np
import random, time
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import math
from node_grid import *
import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
from model.waypoint import Waypoint

MIN_DIST_NONE = 0
MIN_DIST_CPU = 1
MIN_DIST_GPU = 2
   
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
    _start: Waypoint
    _goal: Waypoint
    _velocity_m_s: float
    _img: np.ndarray
    _nodes: NodeGrid
    _goal_reached: bool
    _proc_start: float
    _segmentation_class_cost: list[float]
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
              max_path_size_px: float,
              dist_to_goal_tolerance_px: float,
              class_cost: list[float]
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
        self._class_cost = class_cost
        self._goal_reached = False
       
        
        
    def set_plan_data(self, img: np.ndarray, start: Waypoint, goal: Waypoint, velocity_m_s: float) -> bool:
        self._img = img
        self._start =  (start.x, start.z)
        self._goal = (goal.x, goal.z)
        self._velocity_m_s = velocity_m_s
        
   
    def list_nodes(self) -> list[int2]:
        if self._nodes == None:
            return[]
        return self._nodes.get_node_list()
   
    def search_init(self, cpu_min_dist: int = MIN_DIST_NONE) -> None:
        self._proc_start = time.time()
        self._nodes = NodeGrid(self._width, self._height, (self._start[0], self._start[1]))
        self._nodes.set_search_params((self._min_dist_x, self._min_dist_z),
                                      (self._lower_bound_x, self._lower_bound_z),
                                      (self._upper_bound_x, self._upper_bound_z))
        
        self._nodes.set_physical_params(PhysicalParameters(
                (self._width / self._perception_width_m, self._height / self._perception_height_m),
                self._vehicle_length_m/2,
                math.radians(self._max_steering_angle_deg)))
        self._nodes.set_class_costs(self._class_cost)
        
        if cpu_min_dist == MIN_DIST_CPU:
            self._nodes.check_min_dist()
        elif cpu_min_dist == MIN_DIST_GPU:
            self._nodes.check_min_dist_gpu()
    
    def __sample(self) -> int2:
        return (
            random.randint(0, self._width - 1),
            random.randint(0, self._height - 1)
        )
    
    
    def __dist(self, p1: int2, p2: int2) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.hypot(dx, dy)
    
    def __steer(self, parent: int2, node: int2, max_step_size: float) -> int2:
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
    
    def connect_new_node(self, parent: int2, node: int2, cost: float) -> bool:
        if not self.__check_pos_limits(node):
            return False
        
        self._nodes.add_node(node, parent, cost)
        if (node[0] == parent[0] and node[1] == parent[1]):
            print("cyclic")
        
        return True
    
    def __check_timeout(self) -> bool:
        if self._timeout_ms <= 0:
            return False
        elapsed_time = (time.time() - self._proc_start) * 1000
        return elapsed_time > self._timeout_ms
    
    
    def __check_goal_reached(self, new_node: int2) -> bool:
        dist_to_goal = self.__dist(new_node, self._goal)
        
        if dist_to_goal < self._dist_to_goal_tolerance_px:
            self._goal_node_reached = new_node
            self._goal_reached = True

    
    def loop_rrt(self, kinematic: bool) -> bool:              
        if (self.__check_timeout()):
            return False

        x_rand: int2 = self.__sample()
        x_nearest, dist = self._nodes.find_nearest(x_rand)

        x_new = self.__steer(x_nearest, x_rand, self._max_path_size_px)
        
        res = self.__check_connection(x_nearest, x_new, dist, kinematic)
        if not res.feasible:
            return True
        
        self._nodes.add_node(x_new, x_nearest, res.cost)
        if kinematic:
            self._nodes.set_heading(x_new[0], x_new[1], res.final_heading)

        self.__check_goal_reached(x_new)        
        return True
    
    # def __test_cyclic_ref(self, x: int2):
    #     visited = set()
    #     orig = (x[0], x[1])
    #     p = x
    #     while (p[0] != -1):
    #         key = 1000*x[0] + x[1]
    #         p = self._nodes.get_parent(x)
    #         if key in visited:
    #             print (f"cyclic ref in {orig[0]}, {orig[1]}\n")
    #         visited.add(key)
           
    def __check_connection(self, x_nearest: int2, x_new: int2, dist: float, kinematic: bool) -> GraphConnectionResult:
        if kinematic:
            return self._nodes.check_kinematic_connection(self._img, x_nearest, x_new, self._velocity_m_s)
        else:
            return GraphConnectionResult(
                0.0, self._nodes.check_direct_connection(self._img, x_nearest, x_new), self._nodes.get_cost(x_nearest[0], x_nearest[1]) + dist)            
    
    
    def __equals(self, p1: int2, p2: int2) -> bool:
        return p1[0] == p2[0] and p1[1] == p2[1]
    
    def loop_rrt_star(self, kinematic: bool) -> bool:   
        if (self.__check_timeout()):
            return False

        x_rand: int2 = self.__sample()
        x_nearest, dist = self._nodes.find_nearest(x_rand)

        x_new = self.__steer(x_nearest, x_rand, self._max_path_size_px)
        if (x_nearest[0] == x_new[0] and x_nearest[1] == x_new[1]):
            return True
        
        res = self.__check_connection(x_nearest, x_new, dist, kinematic)
        
        if not res.feasible:
            return True

        x_min = x_nearest
        c_min = res.cost
        heading = res.final_heading
       
        X_near_lst = self._nodes.find_near_nodes(x_new, self._max_path_size_px)
               
        for x_near in X_near_lst:
            res = self.__check_connection(x_near, x_new, dist, kinematic)
     
            if res.feasible and res.cost < c_min:
                x_min = x_near
                c_min = res.cost
                heading = res.final_heading
        
        x_new_parent = x_min
        if not self.connect_new_node(parent=x_new_parent, node=x_new, cost=c_min):
            self._nodes.set_heading(x_new[0], x_new[1], heading)
            #self.__test_cyclic_ref(x_new)
            return True
        
        
        for x_near in X_near_lst:
            if self.__equals(x_near, x_new) or self.__equals(x_near, x_new_parent):
                continue
            
            res = self.__check_connection(x_new, x_near, dist, kinematic)
            
            if res.feasible and res.cost < self._nodes.get_cost(x_near[0], x_near[1]):
                #rewire
                self._nodes.set_parent(x_near, new_parent=x_new)
                self._nodes.set_heading(x_near[0], x_near[1], heading_rad=res.final_heading)

        self.__check_goal_reached(x_new)
        return True
       
 
    def goal_reached(self) -> bool:
        return self._goal_reached
    
    
    def __hermite_interpolation(self, p1: Waypoint, p2: Waypoint) -> list[int2]:     
        curve =[]
        d = Waypoint.distance_between(p1, p2)
        numPoints = int(round(d))
        if numPoints <= 2:
            return [(p1.x, p1.z), (p2.x, p2.z)] 
        a1 = p1.heading - (math.pi / 2)
        a2 = p2.heading - (math.pi / 2)
        
        # Tangent vectors
        tan1 = (d * math.cos(a1), d * math.sin(a1))
        tan2 = (d * math.cos(a2), d * math.sin(a2))

        last_x = -1
        last_z = -1
        
        for i in range (numPoints):
            t = float(i) / (numPoints - 1)

            t2 = t * t
            t3 = t2 * t

            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1.x + h10 * tan1[0] + h01 * p2.x + h11 * tan2[0]
            z = h00 * p1.z + h10 * tan1[1] + h01 * p2.z + h11 * tan2[1]

            if (x < 0 or x >= self._width):
                continue
            if (z < 0 or z >= self._height):
                continue

            cx = int(round(x))
            cz = int(round(z))

            if (cx == last_x and cz == last_z):
                continue

            curve.append((cx, cz))
            last_x = cx
            last_z = cz
        
        return curve

    

    
    
    def get_planned_path(self, interpolate: bool = False):
        nearest_parent, dist = self._nodes.find_nearest((self._goal[0], self._goal[1]))
        
        path = []
        heading = self._nodes.get_heading(nearest_parent[0], nearest_parent[1])
        node = Waypoint(int(nearest_parent[0]), int(nearest_parent[1]), heading)
        while node.x != -1 and node.z != -1:
            path.append(node)
            n = self._nodes.get_parent((node.x, node.z))
            heading = self._nodes.get_heading(n[0], n[1])
            node = Waypoint(n[0], n[1], heading)
        
        path.reverse()
        
        if not interpolate or len(path) < 2:
            return path

        
        intepolate_path = []
        p1 = None
        p2 = None
        for p in path:
            if p1 is None:
                p1 = p
                continue
            p2 = p
            partial_path = self.__hermite_interpolation(p1, p2)
            intepolate_path.extend(partial_path)
            p1 = p2
        
        return intepolate_path
    
    # def get_planned_path(self, interpolate: bool = False):
    #     nearest_parent, dist = self._nodes.find_nearest((self._goal[0], self._goal[1]))
        
    #     path = []
    #     node = (int(nearest_parent[0]), int(nearest_parent[1]))
    #     while node[0] != -1 and node[1] != -1:
    #         path.append(node)
    #         node = self._nodes.get_parent(node)
        
        
    #     if interpolate and len(path) > 1:
    #         x = [p[0] for p in path]
    #         y = [p[1] for p in path]
            
    #         # Generate a parameter t for cubic spline interpolation
    #         t = np.arange(len(path))
            
    #         # Create cubic splines for x and y
    #         cs_x = CubicSpline(t, x)
    #         cs_y = CubicSpline(t, y)
            
    #         # Interpolate with finer granularity
    #         t_fine = np.linspace(0, len(path) - 1, num=10 * len(path))
    #         x_fine = cs_x(t_fine)
    #         y_fine = cs_y(t_fine)
            
    #         # Combine interpolated points into a new path
    #         path = [(int(xf), int(yf)) for xf, yf in zip(x_fine, y_fine)]
        
    #     path.reverse()
    #     return path