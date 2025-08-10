
from pydriveless import Waypoint, MapPose, angle
from pydriveless import CoordinateConverter
from pydriveless import SearchFrame
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
from queue import PriorityQueue
import math, numpy as np

TURNING_COST = 50
DEBUG = False

if DEBUG:
    import cv2

class Node:
    cost: float
    dist: float
    local_pose: Waypoint
    global_pose: MapPose
    parent: 'Node'
    reverse: bool
    
    def __init__(self,
                 parent: 'Node',
                 local_pose: Waypoint,
                 global_pose: MapPose,
                 cost: float, 
                 dist: float,
                 reverse: bool):
        
        self.parent = parent
        self.local_pose = local_pose
        self.cost = cost
        self.dist = dist
        self.global_pose = global_pose
        self.reverse = reverse

    def __gt__(self, other: 'Node'):
        return self.cost > other.cost

    def __lt__(self, other: 'Node'):
        return self.cost < other.cost

    def __eq__(self, other: 'Node'):
        return self.cost == other.cost

class HybridAStar(LocalPlannerExecutor):
    _map_coordinate_converter: CoordinateConverter
    _open_list: PriorityQueue
    _lr: float
    _min_dist: tuple[int, int]
    _dist_to_target_tolerance: float

    MAX_COST = 999999999



    def __init__(self, map_coordinate_converter: CoordinateConverter,
                 max_exec_time_ms: int,
                 dist_to_target_tolerance: float):
        super().__init__("Hybrid A*", max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter
        self._open_list = None
        self._lr = PhysicalParameters.VEHICLE_LENGTH_M / 2
        self._dist_to_target_tolerance = dist_to_target_tolerance
        

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        # ignore
        return False

    #def _compute_dist()

    def _planning_init(self, planning_data: PlanningData) -> bool:
        self._start = planning_data.start()

        self._root = Node(parent=None,
                    local_pose=self._start,
                    global_pose=planning_data.ego_location(),
                    cost=0,
                    dist=planning_data.og().get_distance_to_goal(self._start.x, self._start.z),
                    reverse=False)
        
        if DEBUG:
            gray = np.dot(planning_data.og().get_color_frame()[..., :3], [0.2989, 0.5870, 0.1140])
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            self._og_debug = np.stack((gray,) * 3, axis=-1)

        self._open_list = PriorityQueue()
        self._open_list.put((self._root.cost, self._root))
        self._best_candidate: Node = None
        self._best_cost: float = HybridAStar.MAX_COST
        self._min_dist = planning_data.min_distance()
        self._closed_list: dict[int, Node] = {}
        return super()._planning_init(planning_data)

    def __build_path(self) -> None:
        path: list[Waypoint] = []
        p = self._best_candidate
        
        while p is not None and p is not self._root:
            q = p.local_pose
            path.append(q)
            p = p.parent
        
        path.reverse()
        self._set_planning_result(PlannerResultType.VALID, path)

    def __to_dict_key(p: Waypoint, width: int) -> int:
        return p.z * width + p.x

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        W = planning_data.og().width()

        if  self._open_list.empty():
            self._set_planning_result(PlannerResultType.INVALID_GOAL, path=[])
            return False

        _, curr_point = self._open_list.get(block=False)
        
        
        # chose the best option so far
        if self._best_cost > curr_point.cost:
            self._best_cost = curr_point.cost
            self._best_candidate = curr_point

        # check if we're around the final target
        if curr_point.dist <= self._dist_to_target_tolerance:
            self._best_candidate = curr_point
            self._best_cost = 0
            self.__build_path()
            return False
        
        # check if this node was already visited
        key = HybridAStar.__to_dict_key(curr_point.local_pose, W)
        if key in self._closed_list:
            return True
        
        self._closed_list[key] = curr_point
        # open the node (visit)
        surroundings = self.__compute_free_surroundings(planning_data.og(), planning_data.base_map_conversion_location, curr_point, planning_data.velocity())

        if DEBUG:
            img = np.copy(self._og_debug)
            for p in surroundings:
                img[p.local_pose.z, p.local_pose.x, :] = [255, 0, 0]
            cv2.imwrite('test_output2.png', img)


        for node in surroundings:
            k = HybridAStar.__to_dict_key(node.local_pose, W)
            if k in self._closed_list:
                if node.cost < self._closed_list[k].cost:
                    self._closed_list[k] = node                    
            else:
                # put the children in the priority queue to be processed later
                self._open_list.put((node.dist, node))
        return True
    
    def __curve_model(self, width: int, height: int, map_base_location: MapPose, ego_location: MapPose, velocity_m_s: float, steering_angle: float, steps: int, dt: float = 0.05) -> list[MapPose]:
        """ Generate path from the center of gravity
        """
        v = velocity_m_s
        steer = math.tan(math.radians(steering_angle))
        
        x = ego_location.x
        y = ego_location.y
        heading = ego_location.heading.rad()
        path = []
        local_path = []

        for _ in range (0, steps):
            beta = math.atan(steer / self._lr)
            x += v * math.cos(heading + beta) * dt
            y += v * math.sin(heading + beta) * dt
            heading += v * math.cos(beta) * steer * dt / (2*self._lr)
            next_point = MapPose(x, y, ego_location.z, heading=heading)
            next_point_local = self._map_coordinate_converter.convert(map_base_location, next_point)
            
            if next_point_local.x >= width or next_point_local.x < 0:
                return (path, local_path)

            if next_point_local.z >= height or next_point_local.z < 0:
                return (path, local_path)

            path.append(next_point)
            local_path.append(next_point_local)

        return (path, local_path)

    def __gen_possible_top_paths(self, width: int, height: int, map_base_location: MapPose, node_location: MapPose, velocity_m_s: float, steps: int) -> tuple[list[MapPose], list[Waypoint]]:
        p_top_left = self.__curve_model(width, height,  map_base_location, node_location, velocity_m_s, -PhysicalParameters.MAX_STEERING_ANGLE, steps)
        p_top = self.__curve_model(width, height, map_base_location, node_location, velocity_m_s, 0, steps)
        p_top_right = self.__curve_model(width, height, map_base_location, node_location, velocity_m_s, PhysicalParameters.MAX_STEERING_ANGLE, steps)
        return [p_top_left, p_top, p_top_right]

    def __build_feasible_child_node_list(self, og: SearchFrame, parent: Node, is_turning: bool, map_path: list[MapPose], local_paths: list[Waypoint], local_paths_start: int, local_paths_count: int) -> list[Node]:  
        res = []

        p_parent = parent
        for i in range(local_paths_start, local_paths_count):
            
            p = local_paths[i]
            # leaves at the first unfeasible child node
            if not p.is_checked_as_feasible():
                return res
            
            direction_cost = 0
            if is_turning: 
                direction_cost += TURNING_COST

            dist = og.get_cost(p.x, p.z)
            local_cost = dist + Waypoint.distance_between(parent.local_pose, p) + direction_cost

            n =  Node(local_pose=p,
                global_pose=map_path[i - local_paths_start],
                cost=local_cost,
                dist = dist,
                reverse=False,
                parent=p_parent)
                        
            res.append(n)        
            p_parent = n

        return res

    def __compute_free_surroundings(self, og: SearchFrame, map_base_location: MapPose, node: Node, velocity: float) -> list[Node]:
        tl, t, tr = self.__gen_possible_top_paths(og.width(), og.height(), map_base_location, node.global_pose, velocity, steps=20)

        local_paths = tl[1] + t[1] + tr[1]
        og.check_feasible_path(self._min_dist, local_paths, individual_waypoint_check=True)

        init = 0
        end = len(tl[1])
        nodes_tl = self.__build_feasible_child_node_list(og=og, parent=node, is_turning=True,
                                              map_path=tl[0], local_paths=local_paths,
                                              local_paths_start=init,
                                              local_paths_count=end)

        init = end
        end += len(t[1])
        nodes_t = self.__build_feasible_child_node_list(og=og, parent=node, is_turning=False,
                                              map_path=t[0], local_paths=local_paths,
                                              local_paths_start=init,
                                              local_paths_count=end)

        init = end
        end += len(tr[1])
        nodes_tr = self.__build_feasible_child_node_list(og=og, parent=node, is_turning=True,
                                              map_path=tr[0], local_paths=local_paths,
                                              local_paths_start=init,
                                              local_paths_count=end)
        return nodes_t + nodes_tl + nodes_tr

        
        