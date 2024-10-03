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


DEBUG = True
COLLISION_DETECT = False

class CollisionDetector(DiscreteComponent):
    
    _planning_data_builder: PlanningDataBuilder
    _on_collision_detected_cb: callable
    _planned_path: list[MapPose]
    _map_converter: CoordinateConverter
    _curve_gen: ModelCurveGenerator
    _slam: SLAM
    _on_detect_enable: bool
    
    def __init__(self, 
                 period_ms: int,
                 coordinate_converter: CoordinateConverter,
                 planning_data_builder: PlanningDataBuilder,
                 slam: SLAM,
                 on_collision_detected_cb: callable) -> None:
        super().__init__(period_ms)
        
        self._planning_data_builder = planning_data_builder
        self._on_collision_detected_cb = on_collision_detected_cb
        self._curve_gen = ModelCurveGenerator()
        self._map_converter = coordinate_converter
        self._planned_path = None
        self._slam = slam
        self._on_detect_enable = False
    
    def __check_subpath_feasible(self, path: list[MapPose], r_paths: np.ndarray, start: int) -> bool:
        pos = 0
        for i in range(start, len(path)):
            if not r_paths[pos]:
                return False
            pos += 1
        return True

    def __compute_mean_point_heading(self, p4: MapPose, p3: MapPose, p2: MapPose, p1: MapPose):
        c = 1
        h = MapPose.compute_path_heading(p4, p3)
        
        if p2 is not None:
            h += MapPose.compute_path_heading(p4, p2)
            c += 1

        if p1 is not None:
            h += MapPose.compute_path_heading(p4, p2)
            c += 1
        return h/c
        
    def __compute_path_last_heading(self, path: list[MapPose]) -> float:
        size = len(path)
        if size < 2: return 0
        p4: MapPose = path[size - 1]
        p3: MapPose = None
        p2: MapPose = None
        p1: MapPose = None
        if size > 2: p3 = path[size - 2]
        if size > 3: p2 = path[size - 3]
        if size > 4: p1 = path[size - 4]
        return self.__compute_mean_point_heading(p4, p3, p2, p1)

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


    def ___check_collision(self, og: OccupancyGrid, vel: float) -> bool:
        # TODO: fix - should not be from the current pose but from the final pose on the path
        current_pose = self._slam.estimate_ego_pose()
        watch_pose = self._planned_path[len(self._planned_path) - 1]
        watch_pose.heading = self.__compute_path_last_heading(self._planned_path)
        
        tl, t, tr = self._curve_gen.gen_possible_top_paths(watch_pose, vel, steps=15)
        paths = []
        p1 = self._map_converter.convert_map_path_to_waypoint(current_pose, tl)
        p2 = self._map_converter.convert_map_path_to_waypoint(current_pose, t)
        p3 = self._map_converter.convert_map_path_to_waypoint(current_pose, tr)
        paths.extend(p1)
        paths.extend(p2)
        paths.extend(p3)
        r_paths = og.check_path_feasible(paths)
        
        if DEBUG:
            self.__log_collision_detect(og, p1, p2, p3)
        
        if not self.__check_subpath_feasible(tl, r_paths, start=0):
            return True
        
        if not self.__check_subpath_feasible(t, r_paths, start=len(tl)):
            return True

        if not self.__check_subpath_feasible(tr, r_paths, start=len(tl) + len(t)):
            return True

        return False
    
    def watch_path(self, path: list[MapPose]) -> None:
        self._planned_path = path
        self._on_detect_enable = True

    def _loop(self, dt: float) -> None:
        if not COLLISION_DETECT or not self._on_detect_enable:
            return
        
        if self._planned_path is None or len(self._planned_path) < 5:
            return
        
        planning_data = self._planning_data_builder.build_planning_data()
        
        goal = self._planned_path[len(self._planned_path) - 1]
        goal_point = self._map_converter.convert_map_to_waypoint(planning_data.ego_location, goal)
        
        if goal_point.z >= PhysicalParameters.EGO_UPPER_BOUND.z - 1:
            print("[CD] watch path is too old")
            self._planned_path = None
            return
            
        #og.set_goal_vectorized(goal_point)
        
        if self.___check_collision(planning_data.og, planning_data.velocity):
            print("[CD] collision detected!")
            # un-watch path because it is already invalid in order to avoid multiple alerts for the same problem
            self._planned_path = None
            self._on_detect_enable = False 
            self._on_collision_detected_cb()
        
 