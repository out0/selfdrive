from .angle import angle
from .waypoint import Waypoint
from .search_frame import SearchFrame
from typing import Optional
import ctypes
import os
import numpy as np


class SearchParams:
    _timeout_ms: int
    _max_path_size_px: float
    _distance_to_goal_tolerance_px: float
    _heading_error_tolerance: angle
    _min_distance: tuple[int, int]
    _frame: SearchFrame
    _start: Waypoint
    _goal: Waypoint
    _velocity_m_s: float

    def __init__(
        self,
        timeout_ms: int,
        max_path_size_px: float,
        distance_to_goal_tolerance_px: float,
        heading_error_tolerance: 'angle',
        min_distance: tuple[int, int],
        frame: 'SearchFrame',
        start: 'Waypoint',
        goal: 'Waypoint',
        velocity_m_s: float
    ):
        self._timeout_ms = timeout_ms
        self._max_path_size_px = max_path_size_px
        self._distance_to_goal_tolerance_px = distance_to_goal_tolerance_px
        self._heading_error_tolerance = heading_error_tolerance
        self._min_distance = min_distance
        self._frame = frame
        self._start = start
        self._goal = goal
        self._velocity_m_s = velocity_m_s

    class Builder:
        def __init__(self, start: Waypoint, goal: Waypoint):
            self._timeout_ms = 350
            self._max_path_size_px = 30.0
            self._distance_to_goal_tolerance_px = 20.0
            self._heading_error_tolerance = angle.new_deg(5)
            self._min_distance = (0, 0)
            self._frame = None
            self._start = start
            self._goal = goal
            self._velocity_m_s = 1.0

        def with_timeout(self, timeout_ms: int):
            self._timeout_ms = timeout_ms
            return self

        def with_max_path_size(self, max_path_size_px: float):
            self._max_path_size_px = max_path_size_px
            return self

        def with_distance_to_goal_tolerance(self, distance_px: float):
            self._distance_to_goal_tolerance_px = distance_px
            return self

        def with_heading_error_tolerance(self, heading_error: angle):
            self._heading_error_tolerance = heading_error
            return self

        def with_min_distance(self, min_distance: tuple[int, int]):
            self._min_distance = min_distance
            return self

        def with_frame(self, frame: SearchFrame):
            self._frame = frame
            return self

        def with_velocity(self, velocity_m_s: float):
            self._velocity_m_s = velocity_m_s
            return self

        def build(self) -> 'SearchParams':
            return SearchParams(
                self._timeout_ms,
                self._max_path_size_px,
                self._distance_to_goal_tolerance_px,
                self._heading_error_tolerance,
                self._min_distance,
                self._frame,
                self._start,
                self._goal,
                self._velocity_m_s
            )

    @staticmethod
    def init(start: Waypoint, goal: Waypoint):
        return SearchParams.Builder(start, goal)

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @property
    def max_path_size_px(self) -> float:
        return self._max_path_size_px

    @property
    def distance_to_goal_tolerance_px(self) -> float:
        return self._distance_to_goal_tolerance_px

    @property
    def heading_error_tolerance(self) -> angle:
        return self._heading_error_tolerance

    @property
    def min_distance(self) -> tuple[int, int]:
        return self._min_distance

    @property
    def frame(self) -> Optional[SearchFrame]:
        return self._frame

    @property
    def start(self) -> Waypoint:
        return self._start

    @property
    def goal(self) -> Waypoint:
        return self._goal

    @property
    def velocity_m_s(self) -> float:
        return self._velocity_m_s


class EgoParams:
    _search_frame_dimensions: tuple[int, int]
    _search_frame_physical_dimensions: tuple[float, float]
    _segmentation_class_colors: np.ndarray
    _segmentation_class_costs: np.ndarray

    _ego_lower_bound: tuple[int, int]
    _ego_upper_bound: tuple[int, int]

    _max_steering_angle: angle
    _vehicle_length_m: float
    _max_curvature: float

    _pixel_to_meters_ratio_width: float
    _pixel_to_meters_ratio_height: float
    _meters_to_pixel_ratio_width: float
    _meters_to_pixel_ratio_height: float

    def __init__(self,
                 search_frame_dimensions: tuple[int, int],
                 search_frame_physical_dimensions: tuple[float, float],
                 segmentation_class_colors: np.ndarray,
                 segmentation_class_costs: np.ndarray,
                 ego_lower_bound: tuple[int, int],
                 ego_upper_bound: tuple[int, int],
                 max_steering_angle: angle,
                 vehicle_length_m: float,
                 max_curvature: float,
                 pixel_to_meters_ratio_width: float,
                 pixel_to_meters_ratio_height: float,
                 meters_to_pixel_ratio_width: float,
                 meters_to_pixel_ratio_height: float):
        self._search_frame_dimensions = search_frame_dimensions
        self._search_frame_physical_dimensions = search_frame_physical_dimensions
        self._segmentation_class_colors = segmentation_class_colors
        self._segmentation_class_costs = segmentation_class_costs
        self._ego_lower_bound = ego_lower_bound
        self._ego_upper_bound = ego_upper_bound
        self._max_steering_angle = max_steering_angle
        self._vehicle_length_m = vehicle_length_m
        self._max_curvature = max_curvature
        self._pixel_to_meters_ratio_width = pixel_to_meters_ratio_width
        self._pixel_to_meters_ratio_height = pixel_to_meters_ratio_height
        self._meters_to_pixel_ratio_width = meters_to_pixel_ratio_width
        self._meters_to_pixel_ratio_height = meters_to_pixel_ratio_height
     
    class Builder:
        def __init__(self, search_frame_dimensions: tuple[int, int]):
            self._search_frame_dimensions = search_frame_dimensions
            self._search_frame_physical_dimensions = (-1.0, -1.0)
            self._segmentation_class_colors = []
            self._segmentation_class_costs = []

            self._ego_lower_bound = (-1, -1)
            self._ego_upper_bound = (-1, -1)

            self._max_steering_angle = angle.new_deg(40)
            self._vehicle_length_m = 4.5
            self._max_curvature = 0.35

            self._pixel_to_meters_ratio_width = 1.0
            self._pixel_to_meters_ratio_height = 1.0
            self._meters_to_pixel_ratio_width = 1.0
            self._meters_to_pixel_ratio_height = 1.0

        def with_search_physical_size(self, width_m: float, height_m: float):
            self._search_frame_physical_dimensions = (width_m, height_m)
            return self

        def with_segmentation_class_colors(self, colors: list[tuple[int, int, int]]):
            self._segmentation_class_colors = np.asarray(colors, dtype=np.float32)
            return self

        def with_segmentation_class_costs(self, costs: list[float]):
            self._segmentation_class_costs =  np.asarray(costs, dtype=np.float32)
            return self

        def with_ego_lower_bound(self, bound: tuple[int, int]):
            self._ego_lower_bound = bound
            return self

        def with_ego_upper_bound(self, bound: tuple[int, int]):
            self._ego_upper_bound = bound
            return self

        def with_max_steering_angle(self, max_steering_angle: angle):
            self._max_steering_angle = max_steering_angle
            return self

        def with_vehicle_length(self, vehicle_length_m: float):
            self._vehicle_length_m = vehicle_length_m
            return self

        def with_max_curvature(self, curvature: float):
            self._max_curvature = curvature
            return self

        def build(self) -> 'EgoParams':
            width_px, height_px = self._search_frame_dimensions
            width_m, height_m = self._search_frame_physical_dimensions

            if width_m <= 0 or height_m <= 0:
                width_m = float(width_px)
                height_m = float(height_px)
                self._search_frame_physical_dimensions = (width_m, height_m)

            self._pixel_to_meters_ratio_width = width_m / width_px
            self._pixel_to_meters_ratio_height = height_m / height_px
            self._meters_to_pixel_ratio_width = width_px / width_m
            self._meters_to_pixel_ratio_height = height_px / height_m

            return EgoParams(
                self._search_frame_dimensions,
                self._search_frame_physical_dimensions,
                self._segmentation_class_colors,
                self._segmentation_class_costs,
                self._ego_lower_bound,
                self._ego_upper_bound,
                self._max_steering_angle,
                self._vehicle_length_m,
                self._max_curvature,
                self._pixel_to_meters_ratio_width,
                self._pixel_to_meters_ratio_height,
                self._meters_to_pixel_ratio_width,
                self._meters_to_pixel_ratio_height
            )

    @staticmethod
    def init(search_width: int, search_height: int):
        return EgoParams.Builder((search_width, search_height))

    def new_search_params(self, goal: Waypoint, start: Waypoint = None) -> SearchParams.Builder:
        if start is None:
            w, h = self._search_frame_dimensions
            start = Waypoint(int(0.5 * w), int(0.5 * h), angle.new_rad(0))
        return SearchParams.init(start, goal)

    def new_search_frame(self) -> SearchFrame:
        w, h = self._search_frame_dimensions
        f = SearchFrame(w, h, self._ego_lower_bound, self._ego_upper_bound)
        if self._segmentation_class_costs is not None and len(self._segmentation_class_costs) > 0:
            f.set_class_costs(self._segmentation_class_costs)
        if self._segmentation_class_colors is not None and len(self._segmentation_class_colors) > 0:
            f.set_class_colors(self._segmentation_class_colors)
        return f

    # Accessors
    @property
    def search_frame_dimensions(self) -> tuple[int, int]:
        return self._search_frame_dimensions

    @property
    def search_frame_physical_dimensions(self) -> tuple[float, float]:
        return self._search_frame_physical_dimensions

    @property
    def segmentation_class_colors(self) -> np.ndarray:
        return self._segmentation_class_colors

    @property
    def segmentation_class_costs(self) -> np.ndarray:
        return self._segmentation_class_costs

    @property
    def ego_lower_bound(self) -> tuple[int, int]:
        return self._ego_lower_bound

    @property
    def ego_upper_bound(self) -> tuple[int, int]:
        return self._ego_upper_bound

    @property
    def max_steering_angle(self) -> angle:
        return self._max_steering_angle

    @property
    def vehicle_length_m(self) -> float:
        return self._vehicle_length_m

    @property
    def max_curvature(self) -> float:
        return self._max_curvature

    @property
    def pixel_to_meter_ratio_width(self) -> float:
        return self._pixel_to_meters_ratio_width

    @property
    def pixel_to_meter_ratio_height(self) -> float:
        return self._pixel_to_meters_ratio_height

    @property
    def meter_to_pixel_ratio_width(self) -> float:
        return self._meters_to_pixel_ratio_width

    @property
    def meter_to_pixel_ratio_height(self) -> float:
        return self._meters_to_pixel_ratio_height
