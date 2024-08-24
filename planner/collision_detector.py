from model.discrete_component import DiscreteComponent
from planner.planning_data_builder import PlanningDataBuilder
from vision.occupancy_grid_cuda import OccupancyGrid, GridDirection
from model.map_pose import MapPose
from data.coordinate_converter import CoordinateConverter
from planner.physical_model import ModelCurveGenerator
from model.physical_parameters import PhysicalParameters
from slam.slam import SLAM
import cv2
import numpy as np


DEBUG = True
COLLISION_DETECT = True

class CollisionDetector(DiscreteComponent):
    
    _planning_data_builder: PlanningDataBuilder
    _on_collision_detected_cb: callable
    _planned_path: list[MapPose]
    _map_converter: CoordinateConverter
    _curve_gen: ModelCurveGenerator
    _slam: SLAM
    
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
    
    def __check_subpath_feasible(self, path: list[MapPose], r_paths: np.ndarray) -> bool:
        pos = 0
        for i in range(len(path)):
            if not r_paths[pos]:
                return False
            pos += 1
        return True

    def ___check_collision(self, og: OccupancyGrid, vel: float) -> bool:
        
        current_pose = self._slam.estimate_ego_pose()
        
        tl, t, tr = self._curve_gen.gen_possible_top_paths(current_pose, vel, steps=15)
        paths = []
        paths.extend(self._map_converter.convert_map_path_to_waypoint(current_pose, tl))
        paths.extend(self._map_converter.convert_map_path_to_waypoint(current_pose, t))
        paths.extend(self._map_converter.convert_map_path_to_waypoint(current_pose, tr))
        r_paths = og.check_path_feasible(paths)
        
        if DEBUG:
            f = og.get_color_frame()
            for p in paths:
                f[p.z, p.x, :] = [255, 255, 255]
            cv2.imwrite("colision_frame.png", f)
        
        if not self.__check_subpath_feasible(tl, r_paths):
            return True
        
        if not self.__check_subpath_feasible(t, r_paths):
            return True

        if not self.__check_subpath_feasible(tr, r_paths):
            return True

        
       
        return False
    
    def watch_path(self, path: list[MapPose]) -> None:
        self._planned_path = path

    def _loop(self, dt: float) -> None:
        if not COLLISION_DETECT:
            return
        
        if self._planned_path is None:
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
            # un-watch path because it is already invalid in order to avoid multiple alerts for the same problem
            self._planned_path = None
            self._on_collision_detected_cb()
        
 