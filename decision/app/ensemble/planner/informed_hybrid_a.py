import math
from pydriveless import WorldPose, MapPose, Waypoint, CoordinateConverter, PI
from pydriveless import SearchFrame, angle
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
#from queue import PriorityQueue
import heapq

import numpy as np
import math
from .reeds_shepp import ReedsShepp
import cv2

ORIGINAL_CPU_CHECK = False
DEBUG = True

class Node:
    g_cost: float
    h_cost: float
    f_cost: float
    dist: float
    local_pose: Waypoint
    global_pose: MapPose
    parent: 'Node'
    reverse: bool
    delta_heading: float
    
    def __init__(self,
                 parent: 'Node',
                 local_pose: Waypoint,
                 global_pose: MapPose,
                 g_cost: float, 
                 h_cost: float,
                 f_cost: float,
                 dist: float,
                 delta_heading: float,
                 reverse: bool):
        
        self.parent = parent
        self.local_pose = local_pose
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = f_cost
        self.dist = dist
        self.global_pose = global_pose
        self.reverse = reverse
        self.delta_heading = delta_heading

    def __gt__(self, other: 'Node'):
        return self.f_cost > other.f_cost

    def __lt__(self, other: 'Node'):
        return self.f_cost < other.f_cost

    def __eq__(self, other: 'Node'):
        return self.f_cost == other.f_cost

class IHASConfig:
    veh_dims: tuple[int, int]
    mot_res: float
    n_steer: int
    velocity_m_s: float
    max_steering_rad: float
    
    # cost weights
    H_C: float
    S_C: float
    S_CH_C: float
    B_C: float
    SB_C: float

    def __init__(self,
                veh_dims: tuple[int, int],
                mot_res: float,
                n_steer: int,
                max_steering_rad: float,
                H_C: float,
                S_C: float,
                S_CH_C: float,
                B_C: float,
                SB_C: float):
        self.veh_dims = veh_dims
        self.mot_res = mot_res
        self.n_steer = n_steer
        self.max_steering_rad = max_steering_rad
        self.H_C = H_C
        self.S_C = S_C
        self.S_CH_C = S_CH_C
        self.B_C = B_C
        self.SB_C = SB_C

class InformedHybridA(LocalPlannerExecutor):
    MAX_COST = 999999999

    _map_coordinate_converter: CoordinateConverter
    _width: int
    _height: int
    _map_base_location: MapPose
    _og: SearchFrame
    _min_distance: tuple[int, int]
    _reed_shepp: ReedsShepp
    _goal: Waypoint

    def __init__(self, conv: CoordinateConverter, veh_dims: tuple[int, int], max_exec_time_ms: float = -1, ):
        super().__init__("IHaS", max_exec_time_ms)
        self._map_coordinate_converter = conv
        self._cfg = IHASConfig(
            veh_dims = veh_dims,
            mot_res = 2,
            n_steer = 5,
            max_steering_rad = math.radians(PhysicalParameters.MAX_STEERING_ANGLE),
            # H_C = 3,
            # S_C = 0.5,
            # S_CH_C = 0.5,
            # B_C = 1,
            # SB_C = 10
            H_C = 1,
            S_C = 0,
            S_CH_C = 0,
            B_C = 0,
            SB_C = 0
        )
        self._lr = PhysicalParameters.VEHICLE_LENGTH_M / 2
        

    def __check_local_curves_cpu(self, curves: list[tuple[list[MapPose], list[Waypoint]]]) -> bool:
        # TODO
        feasible_curves = [False for _ in range(len(curves))]
        return feasible_curves
    
    def __check_local_curve_cpu(self, curve: list[Waypoint]) -> bool:
        # TODO
        return False
    
    def __check_local_curves_gpu(self, curves: list[tuple[list[MapPose], list[Waypoint]]]) -> bool:
        all_curves = []
        for c in curves:
            all_curves.extend(c[1])
        
        # massive curve check using the GPU:
        #
        # The original IHaS proposal does not have this improvement, but this vastly increases
        # performance. If you wish to compare to the original proposal, please use __check_local_curves_cpu
        # instead.
        all_curves_res = self._og.check_feasible_path(self._min_distance, all_curves, individual_waypoint_check=True)
        feasible_curves = [True for _ in range(len(curves))]
        if not all_curves_res:
            for i in range(len(curves)):
                feasible = True
                for p in curves[i][1]:
                    if not p.is_checked_as_feasible():
                        feasible = False
                        break
                feasible_curves[i] = feasible
        return feasible_curves

    def __filter_valid_curves(self, curves: list[tuple[list[MapPose], list[Waypoint]]]):
        valid_curves = []
        # filter out curves that are not admissible:
        if ORIGINAL_CPU_CHECK:
            chk = self.__check_local_curves_cpu(curves)
        else:
            chk =self.__check_local_curves_gpu(curves)
        for i in range(len(curves)):
            if chk[i]:
                valid_curves.append(curves[i])
        return valid_curves

    def __to_dict_key(self, local_pose: Waypoint) -> int:
        return local_pose.z * self._width + local_pose.x
    
    def __compute_dist_to_goal(self, local_p: Waypoint) -> float:
        if ORIGINAL_CPU_CHECK:
            return math.sqrt((self._goal.x - local_p.x) ** 2 + (self._goal.z - local_p.z) ** 2)
        else:
            return self._og.get_cost(local_p.x, local_p.z)

    def __compute_costs(self, parent_node: Node, cfg: IHASConfig, local_p: Waypoint, global_p: MapPose) -> tuple[float, float]:
        delta_heading = (global_p.heading - parent_node.global_pose.heading).deg()
        
        change_reverse = parent_node.reverse ^ global_p.reversed

        G_sb = cfg.SB_C * cfg.mot_res if change_reverse else 0
        G_res = cfg.mot_res if global_p.reversed else 0
        
        #G = parent_node.cost + G_res + G_sb + cfg.S_C * abs(delta_heading) + cfg.S_CH_C * abs((delta_heading - parent_node.delta_heading))
        G = parent_node.g_cost + G_res + G_sb + cfg.S_C * abs(delta_heading) + cfg.S_CH_C * abs((delta_heading - parent_node.delta_heading))
        
        dist_to_end = self.__compute_dist_to_goal(local_p)

        H = cfg.H_C * dist_to_end
        
        F = G + H

        return (G, H, F, dist_to_end)

    def __updateSets(self, node: Node, open_set: list, closed_set: dict, cfg: IHASConfig) -> None:
        num_curves = cfg.n_steer
        angles = np.linspace(-cfg.max_steering_rad, cfg.max_steering_rad, num=num_curves, endpoint=True, dtype=np.float32)
        #angles_deg = np.linspace(-math.degrees(cfg.max_steering_rad), math.degrees(cfg.max_steering_rad), num=num_curves, endpoint=True, dtype=np.float32)

        curves_fwd = [self.__curve_model(node.global_pose, cfg.velocity_m_s, a, steps=20) for a in angles]
        curves_bwd = [self.__curve_model(node.global_pose, cfg.velocity_m_s, a, steps=20, reverse=True) for a in angles]

        curves = curves_fwd + curves_bwd

        if DEBUG:
            #f = self._og_debug.copy()
            f = self._og_debug
            for c in curves:
                for p in c[1]:
                    f[p.z, p.x, :] = [128, 0, 128]
            cv2.imwrite("debug.png", f)


        valid_curves = self.__filter_valid_curves(curves)

        key = self.__to_dict_key(node.local_pose)       
        closed_set[key] = node
        
        min_cost = 999999999999999
        min_node = None
        for c in valid_curves:
            last_heading = node.global_pose.heading
            prev_node = node
            for i in range(len(c[0])):
                g, h, f, node_dist = self.__compute_costs(
                    parent_node=node, 
                    cfg=cfg, 
                    local_p=c[1][i],
                    global_p=c[0][i])
                

                n = Node(
                    parent=prev_node,
                    local_pose=c[1][i],
                    global_pose=c[0][i],
                    g_cost=g,
                    f_cost=f,
                    h_cost=h,
                    dist=node_dist,
                    delta_heading=(last_heading - c[0][i].heading).deg(),
                    reverse=c[0][i].reversed                    
                )
                prev_node = n

                heapq.heappush(open_set, (n.f_cost, n))
                # open_set.put()
                if f < min_cost:
                    min_cost = f
                    min_node = n
                #print(f"({c[1][i].x}, {c[1][i].z}) cost: {node_cost}, dist: {node_dist}")

        print(f"best node {min_node.local_pose.x}, {min_node.local_pose.z} with cost {min_node.f_cost} and distance {min_node.dist}")

    def __RS_expansion(self, node: Node, goal: Waypoint, cfg: IHASConfig) -> list[Waypoint]:
        _, _, lx, lz, lh = self._reed_shepp.generation(
            start_pose=node.local_pose,
            goal_pose=goal)
        
        path = []
        for i in range(len(lx)):
            path.append(Waypoint(
                x=lx[i],
                z=lz[i],
                heading=angle.new_rad(lh[i])
            ))
        
        # if DEBUG:
        #     f = self._og_debug.copy()
        #     for p in path:
        #         f[p.z, p.x, :] = [128, 0, 128]
        #     cv2.imwrite("debug.png", f)

        if ORIGINAL_CPU_CHECK:
            valid = self. __check_local_curve_cpu(path)
        else:
            valid = self._og.check_feasible_path(self._min_distance, path, individual_waypoint_check=False)
            

        if valid:
            return path
        else:
            return None

    def __curve_model(self, ego_location: MapPose, velocity_m_s: float, steering_angle_rad: float, steps: int, reverse = False, dt: float = 0.05) -> tuple[list[MapPose], list[Waypoint]]:
        """ Generate path from the center of gravity
        """
        v = velocity_m_s
        steer = math.tan(steering_angle_rad)
        
        x = ego_location.x
        y = ego_location.y
        heading = ego_location.heading.rad()
        if reverse:
            heading += PI
        path = []
        local_path = []

        for _ in range (0, steps):
            beta = math.atan(steer / self._lr)
            x += v * math.cos(heading + beta) * dt
            y += v * math.sin(heading + beta) * dt
            heading += v * math.cos(beta) * steer * dt / (2*self._lr)
            next_point = MapPose(x, y, ego_location.z, heading=heading, reversed=reverse)
            next_point_local = self._map_coordinate_converter.convert(self._map_base_location, next_point)
            
            if next_point_local.x >= self._width or next_point_local.x < 0:
                return (path, local_path)

            if next_point_local.z >= self._height or next_point_local.z < 0:
                return (path, local_path)

            path.append(next_point)
            local_path.append(next_point_local)

        return (path, local_path)
    
    def _planning_init(self, planning_data: PlanningData) -> bool:
        self._og = planning_data.og()
        self._width = self._og.width()
        self._height = self._og.height()
        self._min_distance = planning_data.min_distance()
        self._map_base_location = planning_data.base_map_conversion_location
        self._goal = planning_data.local_goal

        self._reed_shepp = ReedsShepp(
            step=0.1,
            vehicle_length_m=PhysicalParameters.VEHICLE_LENGTH_M,
            max_steering_angle=PhysicalParameters.MAX_STEERING_ANGLE,
            speed=planning_data.velocity()
        )

        if DEBUG:
            gray = np.dot(planning_data.og().get_color_frame()[..., :3], [0.2989, 0.5870, 0.1140])
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            self._og_debug = np.stack((gray,) * 3, axis=-1)

        self._start = planning_data.start()
        self._cfg.velocity_m_s = planning_data.velocity()

        self._root = Node(parent=None,
                    local_pose=self._start,
                    global_pose=planning_data.ego_location(),
                    f_cost=0,
                    g_cost=0,
                    h_cost=self.__compute_dist_to_goal(self._start),
                    dist=planning_data.og().get_distance_to_goal(self._start.x, self._start.z),
                    delta_heading=0.0,
                    reverse=False)

        self._open_list = []
        heapq.heappush(self._open_list, (0, self._root))
        #self._open_list.put((self._root.cost, self._root))
        self._best_candidate: Node = None
        self._best_cost: float = InformedHybridA.MAX_COST
        self._min_dist = planning_data.min_distance()
        self._closed_list: dict[int, Node] = {}
        return super()._planning_init(planning_data)

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        if  len(self._open_list) <= 0:
            self._set_planning_result(PlannerResultType.INVALID_GOAL, path=[])
            return False

        c, node = heapq.heappop(self._open_list)
        self._best_candidate = node
        print (f"processing node: {node.local_pose.x}, {node.local_pose.z} with cost {node.f_cost} heap cost {c}")

        key = self.__to_dict_key(node.local_pose)
        if key in self._closed_list:
            return True
        
        connecting_path = self.__RS_expansion(node, planning_data.local_goal(), self._cfg)
        if connecting_path is not None:
            # I've found a good solution
            self.__build_path(connecting_path)
            return False
        
        self.__updateSets(node, self._open_list, self._closed_list, self._cfg)
        return True
    
    def __build_path(self, connecting_path: list[Waypoint]):
        path: list[Waypoint] = []
        p = self._best_candidate
        
        while p is not None and p is not self._root:
            q = p.local_pose
            path.append(q)
            p = p.parent
        
        path.reverse()
        path.extend(connecting_path)
        self._set_planning_result(PlannerResultType.VALID, path)
        

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        return False


    def draw_curve(self, ego_location: MapPose, velocity_m_s: float, steering_angle: angle, reverse: bool) -> None:
        return self.__curve_model(
            ego_location=ego_location,
            velocity_m_s=velocity_m_s, 
            steering_angle_rad=steering_angle.rad(), 
            steps=20, 
            reverse=reverse
        )
    