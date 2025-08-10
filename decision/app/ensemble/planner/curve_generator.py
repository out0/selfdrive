import math
import numpy as np
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import CoordinateConverter
# from pydriveless import WorldPose, angle
from .. model.physical_paramaters import PhysicalParameters

# EARTH_RADIUS = 6378137.0

# class CoordinateConverter:
    
#     _lat_origin: float
#     _map_pose_origin: MapPose
#     _orig_heading: float
#     _rw: float
#     _rh: float
#     _og_center: Waypoint

#     def __init__(self, world_origin: WorldPose):
#         self._lat_origin = world_origin.lat.rad()
#         self._map_pose_origin = self.convert_world_to_map_pose(world_origin)
#         self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
#         self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT
#         self._og_center = Waypoint(math.floor(PhysicalParameters.OG_WIDTH/2), math.floor(PhysicalParameters.OG_HEIGHT/2))

        
#     def __convert_map_heading_to_compass(h: float) -> float:
#         return (h + 90 + 360) % 360

#     def __convert_compass_to_map_heading(hc: float) -> float:
#         p = (hc - 90 - 360)  % 360
#         if p > 180:
#             return p - 360
#         return p
    
#     def get_relative_map_pose(self, world_pose: WorldPose) -> MapPose:
#         pose = self.convert_world_to_map_pose(world_pose)
#         return pose - self._map_pose_origin
    
#     def get_relative_world_pose(self, map_pose: MapPose) -> WorldPose:
#         m = map_pose + self._map_pose_origin
#         pose = self.convert_map_to_world_pose(m)
#         return pose
    
#     def convert_world_to_map_pose(self, world_pose: WorldPose) -> MapPose:
#         scale = math.cos(self._lat_origin)

#         return MapPose(
#             x=scale * EARTH_RADIUS * world_pose.lon.rad(),
#             y=-scale * EARTH_RADIUS * math.log(math.tan(math.pi * (90 + world_pose.lat.deg()) / 360)),
#             z=world_pose.alt,
#             heading=CoordinateConverter.__convert_compass_to_map_heading(world_pose.compass.deg())
#         )
        
#     def convert_map_to_world_pose(self, map_pose: MapPose) -> WorldPose:
#         scale = math.cos(self._lat_origin)  
#         return WorldPose(
#             lat=360 * math.atan(math.exp(-map_pose.y / (EARTH_RADIUS * scale))) / math.pi - 90,
#             lon=map_pose.x * 180 / (math.pi * EARTH_RADIUS * scale),
#             alt=map_pose.z,
#             heading=CoordinateConverter.__convert_map_heading_to_compass(map_pose.heading)
#         )
    
#     def __build_translation_mat(self, x: float, y: float) -> np.ndarray:
#         return np.array([
#             [1, 0 , 0],
#             [0, 1, 0],
#             [x, y, 1]
#         ])

#     def __build_rotation_mat(self, angle: float) -> np.ndarray:
#         r = math.radians(angle)
#         c = math.cos(r)
#         s = math.sin(r)
    
#         return np.array([
#             [c, s, 0],
#             [-s, c, 0],
#             [0 , 0, 1]
#         ])

#     def __build_resize_mat(self, ratio_x: float, ratio_y) -> np.ndarray:
#         return np.array([
#             [ratio_x, 0, 0],
#             [0, ratio_y, 0],
#             [0 , 0, 1]
#         ])

#     def convert_map_to_Waypoint(self, location: MapPose, target: MapPose) -> Waypoint:
#         m = self.__build_translation_mat(-location.x, -location.y) @\
#             self.__build_rotation_mat(-location.heading) @\
#             self.__build_resize_mat(self._rh, self._rw)
        
#         p = np.array([target.x, target.y, 1]) @ m

#         x = self._og_center.x + p[1]
#         z = self._og_center.z - p[0]
#         return Waypoint(x, z)

#     def  convert_map_path_to_Waypoint(self, location: MapPose, target_list: list[MapPose], copy_heading: bool = False, convert_heading_to_radians: bool = False) -> list[Waypoint]:
#         m = self.__build_translation_mat(-location.x, -location.y) @\
#             self.__build_rotation_mat(-location.heading) @\
#             self.__build_resize_mat(self._rh, self._rw)
        
#         res = []
#         for target in target_list:
#             p = np.array([target.x, target.y, 1]) @ m

#             x = self._og_center.x + p[1]
#             z = self._og_center.z - p[0]
    
#             if copy_heading:
#                 h = target.heading
#                 if convert_heading_to_radians:
#                     h = math.radians(h)
#                 res.append(Waypoint(math.floor(x), math.floor(z), h))
#             else:
#                 res.append(Waypoint(math.floor(x), math.floor(z)))
#         return res

#     def convert_Waypoint_path_to_map_pose(self, location: MapPose, path : list[Waypoint]) -> list[MapPose]:

#         m = self.__build_resize_mat(1/self._rh, 1/self._rw) @\
#             self.__build_rotation_mat(location.heading) @\
#             self.__build_translation_mat(location.x, location.y)
        
#         res = []
#         for Waypoint in path:
#             p = np.array([self._og_center.z - Waypoint.z, Waypoint.x - self._og_center.x, 1]) @ m
#             res.append(MapPose(p[0], p[1], 0, 0))
        
#         return res
        

#     def convert_Waypoint_to_map_pose(self, location: MapPose, target : Waypoint) -> MapPose:

#         p = np.array([self._og_center.z - target.z,
#                       target.x - self._og_center.x, 
#                       1])
    
#         m = self.__build_resize_mat(1/self._rh, 1/self._rw) @\
#             self.__build_rotation_mat(location.heading) @\
#             self.__build_translation_mat(location.x, location.y)
        
#         p = p @ m

#         return MapPose(p[0], p[1], 0, 0)

#     def clip (self, p: Waypoint) -> Waypoint:
#         if p.x < 0:
#             p.x = 0
#         elif p.x >= PhysicalParameters.OG_WIDTH:
#             p.x = PhysicalParameters.OG_WIDTH - 1
#         if p.z < 0:
#             p.z = 0
#         elif p.z >= PhysicalParameters.OG_HEIGHT:
#             p.z = PhysicalParameters.OG_HEIGHT - 1
#         return p
    
#     def Waypoint_convert_ratio (self, w: float, h: float) -> tuple[float, float]:
#         return w * self._rw, h * self._rh


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
        self._L_m = (PhysicalParameters.EGO_LOWER_BOUND[1] - PhysicalParameters.EGO_UPPER_BOUND[1]) / self._pixel_ratio
        self._lr = 0.5 * self._L_m
        self._x_center = math.floor(PhysicalParameters.OG_WIDTH / 2)
        self._z_center = math.floor(PhysicalParameters.OG_HEIGHT / 2)
        self._rw = PhysicalParameters.OG_WIDTH / PhysicalParameters.OG_REAL_WIDTH 
        self._rh = PhysicalParameters.OG_HEIGHT / PhysicalParameters.OG_REAL_HEIGHT  
        self._conv = CoordinateConverter(
            origin=WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0)),
            width=PhysicalParameters.OG_WIDTH, 
            height=PhysicalParameters.OG_HEIGHT, 
            perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT,
            perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH)

        
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
        heading = pos.heading.deg()
        path = []

        for _ in range (0, steps):
            beta = math.degrees(math.atan(steer / self._lr))
            x = x + v * math.cos(math.radians(heading + beta)) * self._delta_t
            y = y + v * math.sin(math.radians(heading + beta)) * self._delta_t
            heading = math.degrees(math.radians(heading) + v * math.cos(math.radians(beta)) * steer * self._delta_t / (2*self._lr))
            next_point = MapPose(x, y, pos.z, heading=angle.new_deg(heading))
            #print(f"new heading for ({next_point.x}, {next_point.y}): {next_point.heading}")
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

    
    def gen_path_Waypoint(self, pos: Waypoint, velocity_meters_per_s: float, steering_angle_deg: float, path_size: float) -> list[Waypoint]:
        """ Generate path from the center of gravity
        """
        v = velocity_meters_per_s
        steer = math.tan(math.radians(steering_angle_deg))
        x, y = self.__simple_change_coordinates_to_map_ref_on_zero_origin_zero_heading(pos)
        
        heading = math.radians(pos.heading)
        dt = 0.1


        path: list[Waypoint] = []
        last_xb = pos.x
        last_zb = pos.z
        ds = v * dt
        size = 0
        beta = math.atan(steer / self._lr)

        while size < path_size:
            x += ds * math.cos(heading + beta)
            y += ds * math.sin(heading + beta)
            heading += ds * math.cos(beta) * steer / (2*self._lr)
           
            xb, zb = self.__simple_change_coordinates_to_bev_ref_on_zero_origin_zero_heading(x, y)
            if last_xb == xb and last_zb == zb:
                continue
            
            next_point = Waypoint(xb, zb, math.degrees(heading)) 
            path.append(next_point)
            last_xb = xb
            last_zb = zb
            size += 1
            
            # size = Waypoint.distance_between(
            #     pos,
            #     next_point
            # )
            
            #print (f"size: {size}")
        
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
        #print (f"current heading: {math.degrees(heading)}, path heading:{math.degrees(path_heading)}, steering: {math.degrees(steering_angle_deg)}")


        ds = v * dt
        total_steps = int(target_distance / ds)
        
        best_end_pos = -1
        best_end_dist = target_distance

        
        for i in range(total_steps):
            steer = math.tan(steering_angle_deg)
            beta = math.atan(steer / self._lr)

            x = x + ds * math.cos(heading + beta)
            y = y + ds * math.sin(heading + beta)
            heading = heading + ds * math.cos(beta) * steer / (2*self._lr)
        

            path_heading = MapPose.compute_path_heading(MapPose(x, y, 0, 0), end_pose)
            steering_angle_deg = np.clip(path_heading - heading, -max_turning_angle, max_turning_angle)            
            #print (f"current heading: {math.degrees(heading)}, path heading:{math.degrees(path_heading)}, steering: {math.degrees(steering_angle_deg)}")

            xb, zb = self.__simple_change_coordinates_to_bev_ref_on_zero_origin_zero_heading(x, y)
            next_point = Waypoint(xb, zb, math.degrees(heading))
            if last_xb == xb and last_zb == zb:
                continue
            
            path.append(next_point)
            
            dist = Waypoint.distance_between(next_point, end)
           # print (f"[Py {i} dist: {dist}]")
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
        p = pos.clone()
        p.heading = pos.heading + 180
        p_bottom_left = self.gen_path_cg(p, velocity_meters_per_s, -PhysicalParameters.MAX_STEERING_ANGLE, steps)
        p_bottom = self.gen_path_cg(p, velocity_meters_per_s, 0, steps)
        p_bottom_right = self.gen_path_cg(p, velocity_meters_per_s, PhysicalParameters.MAX_STEERING_ANGLE, steps)
        return [p_bottom_left, p_bottom, p_bottom_right]
    
