from model.waypoint import Waypoint
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters
import math


class ModelCurveGenerator:
    _L_m: float
    _lr: float
    _pixel_ratio: float
    _ref_local_start: Waypoint
    _delta_t: float
    
    def __init__(self, ego_lower_bound: Waypoint, ego_upper_bound: Waypoint, delta_t: float) -> None:
        self._delta_t = delta_t
        self._pixel_ratio = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        self._L_m = (ego_lower_bound.z - ego_upper_bound.z) / self._pixel_ratio
        self._lr = 0.5 * self.L_m
        self._ref_local_start = Waypoint(
            math.floor(PhysicalParameters.OG_WIDTH / 2),
            ego_upper_bound.z
        )
        
    def get_min_radius() -> float:
        pixel_ratio = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        L_m = (PhysicalParameters.EGO_LOWER_BOUND.z - PhysicalParameters.EGO_UPPER_BOUND.z) / pixel_ratio
        lr = 0.5 * L_m
        steer = math.tan(math.radians(PhysicalParameters.MAX_STEERING_ANGLE))
        return math.atan(steer / lr)
    
    def gen_path(self, pos: MapPose, heading: float, velocity_meters_per_s: float, steering_angle: int, steps: int) -> list[MapPose]:
        v = velocity_meters_per_s
        steer = math.tan(math.radians(steering_angle))
        
        x = pos.x
        y = pos.y
        heading = pos.heading
        path = []

        for _ in range (steps):
            beta = math.degrees(math.atan(steer / self._lr))
            x = x + v * math.cos(math.radians(heading + beta)) * self._delta_t
            y = y + v * math.sin(math.radians(heading + beta)) * self._delta_t
            # if (x > PhysicalParameters.OG_WIDTH or x < 0): break
            # if (y > PhysicalParameters.OG_HEIGHT or y < 0): break
            heading = math.degrees(math.radians(heading) + v * math.cos(math.radians(beta)) * steer * self._delta_t / (2*self._lr))
            next_point = MapPose(x, y, pos.z, heading=heading)
            path.append(next_point)
            
        return path
    
    def gen_possible_top_paths(self, pos: MapPose, velocity_meters_per_s: float, steps: int) -> list[list[MapPose]]:
        heading = pos.heading
        p_top_left = self.gen_path(pos, heading, velocity_meters_per_s, -PhysicalParameters.MAX_STEERING_ANGLE, steps)
        p_top = self.gen_path(pos,  heading, velocity_meters_per_s, 0, steps)
        p_top_right = self.gen_path(pos, heading, velocity_meters_per_s, PhysicalParameters.MAX_STEERING_ANGLE, steps)
        return [p_top_left, p_top, p_top_right]
    
    def gen_possible_bottom_paths(self, pos: MapPose, velocity_meters_per_s: float) -> list[list[MapPose]]:
        heading = 180 - pos.heading
        p_bottom_left = self.gen_path(pos, heading, velocity_meters_per_s, -PhysicalParameters.MAX_STEERING_ANGLE)
        p_bottom = self.gen_path(pos, heading, velocity_meters_per_s, 0)
        p_bottom_right = self.gen_path(pos, heading, velocity_meters_per_s, PhysicalParameters.MAX_STEERING_ANGLE)
        return [p_bottom_left, p_bottom, p_bottom_right]
    