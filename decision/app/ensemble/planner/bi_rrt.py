
from pydriveless import Waypoint, angle
from pydriveless import CoordinateConverter
from pydriveless import SearchFrame
from pydriveless import Interpolator
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
from .bi_rrt_grid import NodeGrid, GridPhysicalParameters, GraphConnectionResult
import math
import numpy as np
import random

int2 = tuple[int, int]
float2 = tuple[float, float]

MIN_DIST_NONE = 0
MIN_DIST_CPU = 1
MIN_DIST_GPU = 2

class BiRRTStar(LocalPlannerExecutor):
    _map_coordinate_converter: CoordinateConverter
    
    _nodes_start: NodeGrid
    _nodes_goal: NodeGrid
    
    _width: int
    _height: int
    _perception_width_m: float
    _perception_height_m: float
    _max_steering_angle_deg: float
    _vehicle_length_m: float
    _min_dist_x: int
    _min_dist_z: int
    _lower_bound_x: int
    _lower_bound_z: int
    _upper_bound_x: int
    _upper_bound_z: int
    _max_path_size_px: float
    _dist_to_goal_tolerance_px: float
    _start: Waypoint
    _class_cost: list[float]
    _velocity_m_s: float
    _kinematic_planning: bool
    
    def __init__(self, map_coordinate_converter: CoordinateConverter,
                 max_exec_time_ms: int, 
                 max_path_size_px: int, 
                 dist_to_goal_tolerance_px: int,
                 class_cost: list[float],
                 kinematic_planning: bool = False
                 ):
        
        super().__init__("Bi-RRT*", max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self._grid = None
        self._max_path_size_px = max_path_size_px
        self._dist_to_goal_tolerance_px = dist_to_goal_tolerance_px
        self._class_cost = class_cost
        self._kinematic_planning = kinematic_planning
   
    def _planning_init(self, planning_data: PlanningData) -> bool:
        self._width = planning_data.og().width()
        self._height = planning_data.og().height()
        self._start = planning_data.start()
        self._perception_width_m = PhysicalParameters.OG_REAL_WIDTH
        self._perception_height_m = PhysicalParameters.OG_REAL_HEIGHT
        self._min_dist_x, self._min_dist_z = planning_data.min_distance()
        self._lower_bound_x, self._lower_bound_z = PhysicalParameters.EGO_LOWER_BOUND
        self._upper_bound_x, self._upper_bound_z = PhysicalParameters.EGO_LOWER_BOUND
        self._velocity_m_s = planning_data.velocity()
        self._vehicle_length_m = PhysicalParameters.VEHICLE_LENGTH_M
        self._max_steering_angle_deg = PhysicalParameters.MAX_STEERING_ANGLE

        # initializing grid from starting node        
        self._nodes_start = self.__initialize_node_grid(self._start, MIN_DIST_GPU)
        
        # initializing grid from goal node
        self._nodes_goal = self.__initialize_node_grid(planning_data.local_goal(), MIN_DIST_GPU)
        return True

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        x_rand: int2 = self.__sample()
        x_new_start = self.__loop_rrt_star_graph(planning_data.og(), self._nodes_start, x_rand, self._kinematic_planning)
        _ = self.__loop_rrt_star_graph(planning_data.og(), self._nodes_goal, x_rand, self._kinematic_planning)
        if self.__check_goal_reached(planning_data.og(), x_new_start, self._kinematic_planning):
            self._set_planning_result(PlannerResultType.VALID, self.get_planned_path(planning_data.local_goal(), True))
            return False
        return True

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        # ignore
        return False

    
    
    # internal methods
    def __sample(self) -> int2:
        return (
            random.randint(1, self._width - 1),
            random.randint(1, self._height - 1)
        )
    def __initialize_node_grid(self, starting_node: Waypoint, cpu_min_dist: int = MIN_DIST_NONE) -> None:
        grid = NodeGrid(self._width, self._height, (starting_node.x, starting_node.z))
        grid.set_search_params((self._min_dist_x, self._min_dist_z),
                                      (self._lower_bound_x, self._lower_bound_z),
                                      (self._upper_bound_x, self._upper_bound_z))
        
        grid.set_physical_params(GridPhysicalParameters(
                (self._width / self._perception_width_m, self._height / self._perception_height_m),
                self._vehicle_length_m/2,
                math.radians(self._max_steering_angle_deg)))
        grid.set_class_costs(self._class_cost)
        grid.set_heading(starting_node.x, starting_node.z, starting_node.heading.rad())
        
        if cpu_min_dist == MIN_DIST_CPU:
            grid.check_min_dist()
        elif cpu_min_dist == MIN_DIST_GPU:
            grid.check_min_dist_gpu()
       
        return grid  
    def __loop_rrt_star_graph(self, og: SearchFrame, grid: NodeGrid, x_rand: int2, kinematic: bool) -> None:
        
        x_nearest, dist = grid.find_nearest(x_rand)
        x_new = self.__steer(x_nearest, x_rand, self._max_path_size_px)
        if (x_nearest[0] == x_new[0] and x_nearest[1] == x_new[1]):
            return None
    
        x_min = None
        c_min = 999999999
        heading = -1
       
        X_near_lst = grid.find_near_nodes(x_new, self._max_path_size_px)
               
        for x_near in X_near_lst:
            res = self.__check_connection(og, grid, x_near, x_new, dist, kinematic)
     
            if res.feasible and res.cost < c_min:
                x_min = x_near
                c_min = res.cost
                heading = res.final_heading
        
        if x_min is None:
            return None

        x_new_parent = x_min
        if not self.__connect_new_node(grid, parent=x_new_parent, node=x_new, cost=c_min):
            #self.__test_cyclic_ref(x_new)
            return None
        
        grid.set_heading(x_new[0], x_new[1], heading)

        for x_near in X_near_lst:
            if self.__equals(x_near, x_new) or self.__equals(x_near, x_new_parent):
                continue
            
            res: GraphConnectionResult = self.__check_connection(og, grid, x_new, x_near, dist, kinematic)
            
            if res.feasible and res.cost < grid.get_cost(x_near[0], x_near[1]):
                #rewire
                grid.set_parent(x_near, new_parent=x_new)
                grid.set_heading(x_near[0], x_near[1], heading_rad=res.final_heading)

        return x_new  
    def __connect_new_node(self, grid: NodeGrid, parent: int2, node: int2, cost: float) -> bool:
        if not self.__check_pos_limits(node):
            return False
        
        grid.add_node(node, parent, cost)
        if (node[0] == parent[0] and node[1] == parent[1]):
            print("cyclic")
        
        return True  
    def __check_pos_limits(self, node: int2) -> bool:
        xc = node[0]
        zc = node[1]
        return xc >= 0 and xc < self._width and zc >= 0 and zc < self._height      
    def __equals(self, p1: int2, p2: int2) -> bool:
        return p1[0] == p2[0] and p1[1] == p2[1]     
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
    def __check_connection(self, og: SearchFrame, grid: NodeGrid, x_nearest: int2, x_new: int2, dist: float, kinematic: bool) -> GraphConnectionResult:
        if kinematic:
            return grid.check_kinematic_connection(og, x_nearest, x_new, self._velocity_m_s)
        else:
            heading = self.__path_heading(x_nearest, x_new)
            return GraphConnectionResult(
                heading, grid.check_direct_connection(og, 
                                                      Waypoint(x_nearest[0], x_nearest[1], angle.new_rad(heading)), 
                                                      Waypoint(x_new[0], x_new[1], angle.new_rad(heading))), grid.get_cost(x_nearest[0], x_nearest[1]) + dist)
            
            
    def __path_heading(self, p1, p2) -> float:
        dz = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        
        if dx == 0 and dz == 0: return 0
        return math.pi/2 - math.atan2(-dz, dx)
    
    def __dist(self, p1: int2, p2: int2) -> float:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.hypot(dx, dy)    
    
    def __check_goal_reached(self, og: SearchFrame, x_new_start: int2, kinematic: bool) -> bool:
        if x_new_start is None:
            return False
        
        x_min = None
        c_min = 999999999
        heading = -1
       
        X_near_lst = self._nodes_goal.find_near_nodes(x_new_start, self._max_path_size_px)
               
        for x_near in X_near_lst:
            dist = self.__dist(x_new_start, x_near)
            res: GraphConnectionResult = self.__check_connection(og, self._nodes_goal, x_near, x_new_start, dist, kinematic)
     
            if res.feasible and res.cost < c_min:
                x_min = x_near
                c_min = res.cost
                heading = res.final_heading
        
        if x_min is None:
            return False
        
        # connect both graphs
        n: int2 = x_min
        p: int2 = x_new_start
        h: float = heading
        last_n: int2 = None

        while (n[0] >= 0 and n[1] >= 0):
            self._nodes_start.add_node(node=n, parent=p, cost=self._nodes_start.get_cost(p[0], p[1]) + self._nodes_start.get_cost(n[0], n[1]))
            self._nodes_start.set_heading(n[0], n[1], h)
            p = n
            last_n = n
            h = self._nodes_goal.get_heading(n[0], n[1]) - math.pi
            if h < 0:
                h = h + 2* math.pi
            n = self._nodes_goal.get_parent(n)

            

        self._goal_reached = True
        self._goal_node_reached = last_n
        return True    

    def get_planned_path(self, goal: Waypoint, interpolate: bool = False):
        nearest_parent, dist = self._nodes_start.find_nearest((goal.x, goal.z))
        
        path = []
        heading = self._nodes_start.get_heading(nearest_parent[0], nearest_parent[1])
        node = (int(nearest_parent[0]), int(nearest_parent[1]), heading)
        while node[0] != -1 and node[1] != -1:
            path.append(node)
            n = self._nodes_start.get_parent((node[0], node[1]))
            heading = self._nodes_start.get_heading(n[0], n[1])
            node = (n[0], n[1], heading)
        
        path.reverse()
        
        if not interpolate or len(path) < 2:
            return path

        
        intepolate_path = []
        p1 = None
        p2 = None
        for p in path:
            if p1 is None:
                p1 = Waypoint(p[0], p[1], angle.new_rad(p[2]))
                continue
            p2 = Waypoint(p[0], p[1], angle.new_rad(p[2]))
            partial_path = Interpolator.hermite(self._width, self._height, p1, p2, True)
            intepolate_path.extend(partial_path)
            p1 = p2
        
        return intepolate_path
    