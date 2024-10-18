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


DEBUG = False
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

    class PathCandidateList:
        path_on_heading: list[Waypoint]
        path_turn_left_heading: list[Waypoint]
        path_turn_right_heading: list[Waypoint]
        path_max_turn_left_heading: list[Waypoint]
        path_max_turn_right_heading: list[Waypoint]

        def __init__(self,
                     path_on_heading: list[Waypoint],
                     path_turn_left_heading: list[Waypoint],
                     path_turn_right_heading: list[Waypoint],
                     path_max_turn_left_heading: list[Waypoint],
                     path_max_turn_right_heading: list[Waypoint]) -> None:
            self.path_on_heading = path_on_heading
            self.path_turn_left_heading = path_turn_left_heading
            self.path_turn_right_heading = path_turn_right_heading
            self.path_max_turn_left_heading = path_max_turn_left_heading
            self.path_max_turn_right_heading = path_max_turn_right_heading

        def __check_path_feasible(self, res: list[bool], start: int, end: int) -> bool:
            for p in res[start: start + end]:
                if not p:
                    return False
            return True

        def get_full_path(self) -> list[Waypoint]:
            all_paths = []
            all_paths.extend(self.path_on_heading)
            all_paths.extend(self.path_turn_left_heading)
            all_paths.extend(self.path_turn_right_heading)
            all_paths.extend(self.path_max_turn_left_heading)
            all_paths.extend(self.path_max_turn_right_heading)
            return all_paths

        def bulk_check_any_fesible(self, og: OccupancyGrid) -> bool:

            all_paths = self.get_full_path()

            l1 = len(self.path_on_heading)
            l2 = len(self.path_turn_left_heading)
            l3 = len(self.path_turn_right_heading)
            l4 = len(self.path_max_turn_left_heading)
            l5 = len(self.path_max_turn_right_heading)

            res = og.check_path_feasible(all_paths, compute_heading=False)

            last = 0
            if self.__check_path_feasible(res, 0, last + l1):
                return True
            
            last += l1
            if self.__check_path_feasible(res, l1, last + l2):
                return True
            
            last += l2
            if self.__check_path_feasible(res, l2, last + l3):
                return True
            
            last += l3
            if self.__check_path_feasible(res, l3, last + l4):
                return True
            
            last += l4
            if self.__check_path_feasible(res, l4, last + l5):
                return True

            return False

    def __init__(self,
                 period_ms: int,
                 coordinate_converter: CoordinateConverter,
                 planning_data_builder: PlanningDataBuilder,
                 slam: SLAM,
                 on_collision_detected_cb: callable,
                 with_telemetry: bool = LOG) -> None:
        super().__init__(period_ms)

        self.__planning_data_builder = planning_data_builder
        self.__on_collision_detected_cb = on_collision_detected_cb
        self.__watch_path = None
        self.__watch_target = None
        self.__coord_converter = coordinate_converter
        self.__kinematics_model = ModelCurveGenerator()
        self.__slam = slam
        self.__on_detect_enable = False
        self.__with_telemetry = with_telemetry

    def watch_path(self, path: list[MapPose]) -> None:
        self.__watch_path = path
        self.__watch_target = self.__compute_target_on_watch_path(path)
        self.__on_detect_enable = True

    def __log_collision_detect(self, og: OccupancyGrid, path: list[Waypoint]) -> None:

        f = og.get_color_frame()

        for p in path:
            if (p.z < 0 or p.z > og.height()):
                continue
            if (p.x < 0 or p.x > og.width()):
                continue
            f[p.z, p.x, :] = [255, 255, 255]

        cv2.imwrite("colision_frame.png", f)
        with open("collision_path.log", "w") as log:
            data = []
            for p in path:
                data.append(str(p))
            log.write(json.dumps(data))

    def __compute_target_on_watch_path(self, path: list[MapPose]) -> MapPose:
        if len(path) < 3:
            return None

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

    # def __project_and_clip_outbound(self,  og: OccupancyGrid, location: MapPose, path: list[MapPose]) -> list[Waypoint]:
    #     res: list[Waypoint] = []
    #     for p in path:
    #         wp: Waypoint = self.__coord_converter.convert_map_to_waypoint(
    #             location, p)
    #         if wp.x < 0 or wp.x >= og.width():
    #             continue
    #         if wp.z < 0 or wp.z >= og.height():
    #             continue
    #         res.append(wp)
    #     return res

    def __check_collision(self, og: OccupancyGrid, location: MapPose, vel: float) -> tuple[bool, list[Waypoint]]:
        # TODO: fix - should not be from the current pose but from the final pose on the path

        half_steer = PhysicalParameters.MAX_STEERING_ANGLE / 2
        
        h =  self.__watch_target.heading

        path_on_heading = self.__kinematics_model.gen_path_cg(
            self.__watch_target, vel, h, 15)
        path_turn_left_heading = self.__kinematics_model.gen_path_cg(
            self.__watch_target, vel, h-half_steer, 15)
        path_turn_right_heading = self.__kinematics_model.gen_path_cg(
            self.__watch_target, vel, h+half_steer, 15)
        path_max_turn_left_heading = self.__kinematics_model.gen_path_cg(
            self.__watch_target, vel, h-PhysicalParameters.MAX_STEERING_ANGLE, 15)
        path_max_turn_right_heading = self.__kinematics_model.gen_path_cg(
            self.__watch_target, vel, h+PhysicalParameters.MAX_STEERING_ANGLE, 15)
        
        path_on_heading = self.__coord_converter.convert_map_path_to_waypoint(location, path_on_heading, True, True)
        path_turn_left_heading = self.__coord_converter.convert_map_path_to_waypoint(location, path_turn_left_heading, True, True)
        path_turn_right_heading = self.__coord_converter.convert_map_path_to_waypoint(location, path_turn_right_heading, True, True)
        path_max_turn_left_heading = self.__coord_converter.convert_map_path_to_waypoint(location, path_max_turn_left_heading, True, True)
        path_max_turn_right_heading = self.__coord_converter.convert_map_path_to_waypoint(location, path_max_turn_right_heading, True, True)

        paths = CollisionDetector.PathCandidateList(
            path_on_heading=path_on_heading,
            path_turn_left_heading=path_turn_left_heading,
            path_turn_right_heading=path_turn_right_heading,
            path_max_turn_left_heading=path_max_turn_left_heading,
            path_max_turn_right_heading=path_max_turn_right_heading,
        )

        if DEBUG:
            all_paths = paths.get_full_path()
            self.__log_collision_detect(og, all_paths)

        if paths.bulk_check_any_fesible(og):
            return False, None

        return True, paths.get_full_path()

    def __check_path_is_too_old(self, location: MapPose) -> bool:

        target_waypoint = self.__coord_converter.convert_map_to_waypoint(
            location, self.__watch_target)
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

        collision, paths = self.__check_collision(
            planning_data.og, planning_data.ego_location, planning_data.velocity)

        if collision:
            print("[CD] collision detected!")

            if self.__with_telemetry:
                Telemetry.log_collision(CollisionReport(
                    collision_time=time.time(),
                    primitives=paths,
                    watch_path=self.__watch_path,
                    watch_target=self.__watch_target,
                    ego_location=planning_data.ego_location
                ), planning_data.bev)

            # un-watch path because it is already invalid in order to avoid multiple alerts of the same problem
            self.__watch_path = None
            self.__watch_target = None
            self.__on_detect_enable = False

            # call the callback to warn the controller about the collision
            self.__on_collision_detected_cb()
            # time.sleep(2)
#            exit(1)
