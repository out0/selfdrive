from model.discrete_component import DiscreteComponent
from planner.planning_data_builder import PlanningDataBuilder
from vision.occupancy_grid_cuda import OccupancyGrid, GridDirection
from model.map_pose import MapPose
from data.coordinate_converter import CoordinateConverter
from planner.physical_model import ModelCurveGenerator
from model.physical_parameters import PhysicalParameters
from slam.slam import SLAM
from model.waypoint import Waypoint
import cv2
import numpy as np
import json
from utils.telemetry import Telemetry
from model.planning_data import CollisionReport
import time


DEBUG = True
COLLISION_DETECT = False
LOG = True

class CollisionDetector(DiscreteComponent):
    
    __planning_data_builder: PlanningDataBuilder
    __on_collision_detected_cb: callable
    __watch_path: list[MapPose]
    __watch_target: MapPose
    __coord_converter: CoordinateConverter
    __kinematics_model: ModelCurveGenerator
    __slam: SLAM
    __on_detect_enable: bool
    
    def __init__(self, 
                 period_ms: int,
                 coordinate_converter: CoordinateConverter,
                 planning_data_builder: PlanningDataBuilder,
                 slam: SLAM,
                 on_collision_detected_cb: callable) -> None:
        super().__init__(period_ms)
        
        self.__planning_data_builder = planning_data_builder
        self.__on_collision_detected_cb = on_collision_detected_cb
        self.__watch_path = None
        self.__watch_target = None
        self.__coord_converter = coordinate_converter
        self.__kinematics_model = ModelCurveGenerator()
        self.__slam = slam
        self.__on_detect_enable = False
    
    def watch_path(self, path: list[MapPose]) -> None:
        self.__watch_path = path
        self.__watch_target = self.__compute_target_on_watch_path(path)
        self.__on_detect_enable = True
    
    def __check_subpath_feasible(self, r_paths: np.ndarray, start: int, end: int) -> bool:
        for i in range(start, end):
            if not r_paths[i]:
                return False
        return True

    
    def __log_collision_detect(self, og: OccupancyGrid, p1: list[Waypoint], p2: list[Waypoint], p3: list[Waypoint]) -> None:
    
            f = og.get_color_frame()

            for path in [p1, p2, p3]:           
                for p in path:
                    if (p.z < 0 or p.z > og.height()): continue
                    if (p.x < 0 or p.x > og.width()): continue
                    f[p.z, p.x, :] = [255, 255, 255]
            
            cv2.imwrite("colision_frame.png", f)
            with open("collision_path.log", "w") as log:
                data = {}
                data['left'] = list(map(lambda p: str(p), p1))
                data['center'] = list(map(lambda p: str(p), p2))
                data['right'] = list(map(lambda p: str(p), p3))
                log.write(json.dumps(data))

    def __generate_primitives(self, og:OccupancyGrid, location: MapPose, target: MapPose, vel: float) -> tuple[list[Waypoint], list[Waypoint], list[Waypoint]]:
        tl, t, tr = self.__kinematics_model.gen_possible_top_paths(target, vel, steps=15)
        path_max_left = self.__coord_converter.convert_map_path_to_waypoint(location, tl)
        path_center = self.__coord_converter.convert_map_path_to_waypoint(location, t)
        path_max_right = self.__coord_converter.convert_map_path_to_waypoint(location, tr)
        
        return (path_max_left, path_center, path_max_right)
    
    def __compute_target_on_watch_path(self, path: list[MapPose]) -> MapPose:
        if len(path) < 3: return None
        
        l = len(path) - 1
        p0 = path[l]
        p1 = path[l-1]
        p2 = path[l-2]
        p3 = path[l-3]
        
        # compute the heading by mean
        head_0 = MapPose.compute_path_heading(p3, p0)
        head_1 = MapPose.compute_path_heading(p2, p0)
        head_2 = MapPose.compute_path_heading(p1, p0)
        
        target = p0.clone()
        target.heading = (head_0 + head_1 + head_2) / 3
        return target
        

    def ___check_collision(self, og: OccupancyGrid, location: MapPose, vel: float) -> tuple[bool, list[Waypoint]]:
        # TODO: fix - should not be from the current pose but from the final pose on the path
        
               
        p1, p2, p3 = self.__generate_primitives(og, location, self.__watch_target, vel)
        
        paths: list[Waypoint] = []
        paths.extend(p1)
        paths.extend(p2)
        paths.extend(p3)
        r_paths = og.check_path_feasible(paths)
        
        if self.__check_subpath_feasible(r_paths, start=0, end=len(p1)):
            return False, None
        
        if self.__check_subpath_feasible(r_paths, start=len(p1), end=len(p1) + len(p2)):
            return False, None

        if self.__check_subpath_feasible(r_paths, start=len(p2), end=len(paths)):
            return False, None

        if DEBUG:
            self.__log_collision_detect(og, p1, p2, p3)


        return True, paths
    
    def __check_path_is_too_old(self, location: MapPose) -> bool:
        
        target_waypoint = self.__coord_converter.convert_map_to_waypoint(location, self.__watch_target)
        return target_waypoint.z >= PhysicalParameters.EGO_UPPER_BOUND.z - 1
   

    def _loop(self, dt: float) -> None:
        if not COLLISION_DETECT or not self.__on_detect_enable:
            return
        
        if self.__watch_path is None or len(self.__watch_path) < 5:
            return
        
        if self.__watch_target is None:
            return
        
        location = self.__slam.estimate_ego_pose()
        planning_data = self.__planning_data_builder.build_planning_data()
        
        if self.__check_path_is_too_old(location):
            print("[CD] watch path is too old")
            self._planned_path = None
            return

        
        collision, paths = self.___check_collision(planning_data.og, location, planning_data.velocity)
        
        if collision:
            print("[CD] collision detected!")

            if LOG:
                Telemetry.log_collision(CollisionReport(
                    collision_time=time.time(),
                    primitives=paths,
                    watch_path=self.__watch_path,
                    watch_target=self.__watch_target
                ))

            # un-watch path because it is already invalid in order to avoid multiple alerts of the same problem
            self.__watch_path = None
            self.__watch_target = None
            self.__on_detect_enable = False
            
            # call the callback to warn the controller about the collision
            self.__on_collision_detected_cb()
        
 