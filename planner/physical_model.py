from model.waypoint import Waypoint
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters
from data.coordinate_converter import CoordinateConverter
from model.world_pose import WorldPose
import math
import numpy as np


class ModelCurveGenerator:
    _L_m: float
    _lr: float
    _delta_t: float
    _x_center: int
    _z_center: int
    _rw: float
    _rh: float    
    _conv: CoordinateConverter

    
    def __init__(self, delta_t: float = 0.05) -> None:
        self._delta_t = delta_t
        self._pixel_ratio = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        self._L_m = (PhysicalParameters.EGO_LOWER_BOUND.z - PhysicalParameters.EGO_UPPER_BOUND.z) / self._pixel_ratio
        self._lr = 0.5 * self._L_m
        self._x_center = math.floor(PhysicalParameters.OG_WIDTH / 2)
        self._z_center = math.floor(PhysicalParameters.OG_HEIGHT / 2)
        self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
        self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT  
        self._conv = CoordinateConverter(WorldPose(0, 0, 0, 0))

        
    def get_lr(self) -> float:
        return self._lr
    
    @classmethod
    def get_min_radius(cls) -> float:
        pixel_ratio = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        L_m = (PhysicalParameters.EGO_LOWER_BOUND.z - PhysicalParameters.EGO_UPPER_BOUND.z) / pixel_ratio
        lr = 0.5 * L_m
        steer = math.tan(math.radians(PhysicalParameters.MAX_STEERING_ANGLE))
        return math.atan(steer / lr)
    
    def gen_path_cg(self, pos: MapPose, velocity_meters_per_s: float, steering_angle: int, steps: int) -> list[MapPose]:
        """ Generate path from the center of gravity
        """
        v = velocity_meters_per_s
        steer = math.tan(math.radians(steering_angle))
        
        x = pos.x
        y = pos.y
        heading = pos.heading
        path = []

        for _ in range (0, steps):
            beta = math.degrees(math.atan(steer / self._lr))
            x = x + v * math.cos(math.radians(heading + beta)) * self._delta_t
            y = y + v * math.sin(math.radians(heading + beta)) * self._delta_t
            heading = math.degrees(math.radians(heading) + v * math.cos(math.radians(beta)) * steer * self._delta_t / (2*self._lr))
            next_point = MapPose(x, y, pos.z, heading=heading)
            print(f"new heading for ({next_point.x}, {next_point.y}): {next_point.heading}")
            path.append(next_point)
            
        return path
    
    def gen_path_cg2(self, pos: MapPose, velocity_meters_per_s: float, steering_angle_deg: float, steps: int) -> list[MapPose]:
        """ Generate path from the center of gravity
        """
        v = velocity_meters_per_s
        dt = 0.5
        steer = math.tan(math.radians(steering_angle_deg))
        
        x = pos.x
        y = pos.y
        heading = pos.heading
        path = []

        for _ in range (0, steps):
            beta = math.degrees(math.atan(steer / self._lr))
            x = x + v * math.cos(math.radians(heading + beta)) * dt
            y = y + v * math.sin(math.radians(heading + beta)) * dt
            heading = math.degrees(math.radians(heading) + v * math.cos(math.radians(beta)) * steer * dt / (2*self._lr))
            next_point = MapPose(x, y, pos.z, heading=heading)
            path.append(next_point)
            
        return path
    
    def __simple_change_coordinates_to_map_ref_on_zero_origin_zero_heading(self, pos: Waypoint) -> tuple[float, float]:
        x = (self._x_center - pos.z) / self._rw
        y = (pos.x - self._z_center) / self._rh
        return x, y

    def __simple_change_coordinates_to_bev_ref_on_zero_origin_zero_heading(self, x: float, y: float) -> tuple[int, int]:
        xb = math.floor(self._z_center + self._rh * y)
        zb = math.floor(self._x_center - self._rw * x)
        return xb, zb

    
    def gen_path_waypoint(self, pos: Waypoint, velocity_meters_per_s: float, steering_angle_deg: float, path_size: float) -> list[Waypoint]:
        """ Generate path from the center of gravity
        """
        v = velocity_meters_per_s
        steer = math.tan(math.radians(steering_angle_deg))
        x, y = self.__simple_change_coordinates_to_map_ref_on_zero_origin_zero_heading(pos)
        heading = pos.heading
        dt = 0.1

        path: list[Waypoint] = []
        last_xb = pos.x
        last_zb = pos.z
        ds = v * dt
        size = 0

        while size < path_size:
            beta = math.degrees(math.atan(steer / self._lr))
            x = x + ds * math.cos(math.radians(heading + beta))
            y = y + ds * math.sin(math.radians(heading + beta))
            heading = math.degrees(math.radians(heading) + ds * math.cos(math.radians(beta)) * steer / (2*self._lr))            
            
            xb, zb = self.__simple_change_coordinates_to_bev_ref_on_zero_origin_zero_heading(x, y)
            if last_xb == xb and last_zb == zb:
                continue
            
            next_point = Waypoint(xb, zb, heading)            
            path.append(next_point)
            last_xb = xb
            last_zb = zb
            # size += 1
            
            size = Waypoint.distance_between(
                pos,
                next_point
            )
            
            print (f"size: {size}")
        
        return path
    
    def connect_nodes_with_path(self, start: Waypoint, end: Waypoint, velocity_meters_per_s: float) -> list[Waypoint]:
        """ Generate path from the center of gravity
        """
        v = velocity_meters_per_s
        
        x, y = self.__simple_change_coordinates_to_map_ref_on_zero_origin_zero_heading(start)
        xe, ye = self.__simple_change_coordinates_to_map_ref_on_zero_origin_zero_heading(end)
        end_pose = MapPose(xe, ye, 0, 0)
        
        dt = 0.1
        path: list[Waypoint] = []
        
        last_xb = start.x
        last_zb = start.z
        
        dx = end.x - start.x
        dz = end.z - start.z
        target_distance = np.hypot(dx, dz)
        max_turning_angle = math.radians(PhysicalParameters.MAX_STEERING_ANGLE)
        heading = math.radians(start.heading)
        
        path_heading = MapPose.compute_path_heading(MapPose(x, y, 0, 0), end_pose)
        steering_angle_deg = np.clip(path_heading - heading, -max_turning_angle, max_turning_angle)
        print (f"current heading: {math.degrees(heading)}, path heading:{math.degrees(path_heading)}, steering: {math.degrees(steering_angle_deg)}")


        ds = v * dt
        total_steps = int(target_distance / ds)
        
        best_end_pos = -1
        best_end_dist = target_distance
        
        for _ in range(total_steps):
            steer = math.tan(steering_angle_deg)
            beta = math.atan(steer / self._lr)

            x = x + ds * math.cos(heading + beta)
            y = y + ds * math.sin(heading + beta)
            heading = heading + ds * math.cos(beta) * steer / (2*self._lr)

            path_heading = MapPose.compute_path_heading(MapPose(x, y, 0, 0), end_pose)
            steering_angle_deg = np.clip(path_heading - heading, -max_turning_angle, max_turning_angle)            
            print (f"current heading: {math.degrees(heading)}, path heading:{math.degrees(path_heading)}, steering: {math.degrees(steering_angle_deg)}")

            xb, zb = self.__simple_change_coordinates_to_bev_ref_on_zero_origin_zero_heading(x, y)
            next_point = Waypoint(xb, zb, heading)
            if last_xb == xb and last_zb == zb:
                continue
            
            path.append(next_point)
            
            dist = Waypoint.distance_between(next_point, end)
            if best_end_dist > dist:
                best_end_dist = dist
                best_end_pos = len(path)
            
            last_xb = xb
            last_zb = zb
        
        return path[0:best_end_pos]

    
    def gen_possible_top_paths(self, pos: MapPose, velocity_meters_per_s: float, steps: int = 20) -> list[list[MapPose]]:
        p_top_left = self.gen_path_cg(pos, velocity_meters_per_s, -PhysicalParameters.MAX_STEERING_ANGLE, steps)
        p_top = self.gen_path_cg(pos, velocity_meters_per_s, 0, steps)
        p_top_right = self.gen_path_cg(pos, velocity_meters_per_s, PhysicalParameters.MAX_STEERING_ANGLE, steps)
        return [p_top_left, p_top, p_top_right]
    
    def gen_possible_bottom_paths(self, pos: MapPose, velocity_meters_per_s: float, steps: int = 20) -> list[list[MapPose]]:
        p = pos + 0
        p.heading = 180 - pos.heading
        p_bottom_left = self.gen_path_cg(p, velocity_meters_per_s, -PhysicalParameters.MAX_STEERING_ANGLE, steps)
        p_bottom = self.gen_path_cg(p, velocity_meters_per_s, 0, steps)
        p_bottom_right = self.gen_path_cg(p, velocity_meters_per_s, PhysicalParameters.MAX_STEERING_ANGLE, steps)
        return [p_bottom_left, p_bottom, p_bottom_right]
    