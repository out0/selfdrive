from model.waypoint import Waypoint
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters
import math


class ModelCurveGenerator:
    _L_m: float
    _lr: float
    _delta_t: float
    
    def __init__(self, delta_t: float = 0.05) -> None:
        self._delta_t = delta_t
        self._pixel_ratio = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
        self._L_m = (PhysicalParameters.EGO_LOWER_BOUND.z - PhysicalParameters.EGO_UPPER_BOUND.z) / self._pixel_ratio
        self._lr = 0.5 * self._L_m
        
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
    
    def gen_path_waypoint(self, pos: Waypoint, velocity_meters_per_s: float, steering_angle_deg: float, path_size: float) -> list[Waypoint]:
        """ Generate path from the center of gravity
        """
        self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
        self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT  
        v = velocity_meters_per_s
        steer = math.tan(math.radians(steering_angle_deg))
        
        x = (128 - pos.z) / self._rw
        y = (pos.x - 128) / self._rh
        heading = 0
        path = []
        
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
            xb = math.floor(128 + self._rh * y)
            zb = math.floor(128 - self._rw * x)
                        
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
    
    # def gen_path_waypoint(self, pos: Waypoint, velocity_meters_per_s: float, steering_angle_deg: float, path_size: float) -> list[Waypoint]:
    #     """ Generate path from the center of gravity
    #     """
    #     self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
    #     self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT  
    #     v = velocity_meters_per_s
    #     steer = math.tan(math.radians(steering_angle_deg))
        
    #     x = (128 - pos.z) / self._rw
    #     y = (pos.x - 128) / self._rh
    #     heading = 0
    #     path = []
        
    #     dt = 0.1
    #     #steps = math.floor(path_size / (dt * velocity_meters_per_s))
    #     steps = math.floor(path_size / v)
    #     #dt = 1/velocity_meters_per_s

    #     path: list[Waypoint] = []
        
    #     last_xb = pos.x
    #     last_zb = pos.z

    #     ds = v * dt
    #     for _ in range (0, steps):            
    #         beta = math.degrees(math.atan(steer / self._lr))
    #         x = x + ds * math.cos(math.radians(heading + beta))
    #         y = y + ds * math.sin(math.radians(heading + beta))
    #         heading = math.degrees(math.radians(heading) + ds * math.cos(math.radians(beta)) * steer / (2*self._lr))            
    #         xb = math.floor(128 + self._rh * y)
    #         zb = math.floor(128 - self._rw * x)
                        
    #         if last_xb == xb and last_zb == zb:
    #             continue
            
    #         next_point = Waypoint(xb, zb, heading)            
    #         path.append(next_point)
        
    #     return path

    
    # def gen_path_cg_by_driving_time(self, pos: MapPose, steering_angle: int, velocity_meters_per_s: float, driving_time: float, dt: float) -> list[MapPose]:
    #     """ Generate path from the center of gravity
    #     """
    #     steer = math.tan(math.radians(steering_angle))
    #     beta = math.degrees(math.atan(steer / self._lr))

    #     x = pos.x
    #     y = pos.y
    #     v = velocity_meters_per_s
    #     heading = pos.heading
    #     path = []
        
    #     steps = math.ceil(driving_time / dt)

    #     for _ in range (steps):
    #         x = x + v * math.cos(math.radians(heading + beta)) * self._delta_t
    #         y = y + v * math.sin(math.radians(heading + beta)) * self._delta_t
    #         heading = math.degrees(math.radians(heading) + v * math.cos(math.radians(beta)) * steer * self._delta_t / (2*self._lr))
    #         next_point = MapPose(x, y, pos.z, heading=heading)
    #         path.append(next_point)
            
    #     return path
    
        
       
    
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
    