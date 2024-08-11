from model.map_pose import MapPose
from model.waypoint import Waypoint
import numpy as np
from enum import Enum
from io import StringIO
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters


class PrePlanningData:
    top_frame:  np.ndarray
    bottom_frame: np.ndarray
    left_frame: np.ndarray
    right_frame: np.ndarray
    ego_location: MapPose
    velocity: float

    def __init__(self, ego_location: MapPose, velocity: float, top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self.ego_location = ego_location
        self.velocity = velocity
        self.top_frame = top
        self.bottom_frame = bottom
        self.left_frame = left
        self.right_frame = right

    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"ego_location:{self.ego_location},velocity:{self.velocity},top:{self.__frame_shape_to_str(self.top_frame)},left:{self.__frame_shape_to_str(self.left_frame)},right:{self.__frame_shape_to_str(self.right_frame)},bottom:{self.__frame_shape_to_str(self.bottom_frame)}"

class PlanningData:
    bev: np.ndarray
    og: OccupancyGrid
    ego_location: MapPose
    velocity: float
    goal: MapPose
    next_goal: MapPose

    def __init__(self, bev: np.ndarray, ego_location: MapPose, velocity: float, goal: MapPose, next_goal: MapPose) -> None:
        self.bev = bev
        self.og = OccupancyGrid(
            frame=bev,
            minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
            upper_bound=PhysicalParameters.EGO_UPPER_BOUND
        )
        self.ego_location = ego_location
        self.velocity = velocity
        self.goal = goal
        self.next_goal = next_goal

    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"ego_location:{self.ego_location},velocity:{self.velocity},bev:{self.__frame_shape_to_str(self.bev)},goal:{self.goal},next_goal:{self.next_goal}"

class PlannerResultType(Enum):
    NONE =0
    VALID = 1
    INVALID_START = 2
    INVALID_GOAL = 3
    INVALID_PATH = 4
    TOO_CLOSE = 5
    pass


class PlanningResult:
    result_type: PlannerResultType
    path: list[Waypoint]
    timeout: bool
    planner_name: str
    total_exec_time_ms: int
    local_start: Waypoint
    local_goal: Waypoint
    goal_direction: int
    ego_location: MapPose
    map_goal: MapPose
    map_next_goal: MapPose
    
    def __init__(self) -> None:
        self.result_type = PlannerResultType.NONE
        self.path = None
        self.timeout = False
        self.planner_name = "-"
        self.total_exec_time_ms = 0
        self.local_start = None
        self.local_goal  = None
        self.goal_direction = 0
        self.ego_location = None
        self.map_goal = None
        self.map_next_goal = None

    def __str__(self) -> str:
        str = StringIO()

        if self.result_type == PlannerResultType.NONE:
            return "-"
        elif self.result_type == PlannerResultType.VALID:
            return f"({self.ego_location} -> {self.map_goal}) valid plan to goal, waypoint: {self.local_goal}, timeout: {self.timeout}"
        elif self.result_type == PlannerResultType.INVALID_START:
            return f"({self.ego_location} -> {self.map_goal}) INVALID **start**, waypoint: {self.local_goal}, timeout: {self.timeout}"
        elif self.result_type == PlannerResultType.INVALID_PATH:
            return f"({self.ego_location} -> {self.map_goal}) INVALID **plan**, waypoint: {self.local_goal}, timeout: {self.timeout}"
        elif self.result_type == PlannerResultType.INVALID_GOAL:
            return f"({self.ego_location} -> {self.map_goal}) INVALID **goal**, waypoint: {self.local_goal}, timeout: {self.timeout}"
        return str.getvalue()
    
    def clone(self) -> 'PlanningResult':
        res = PlanningResult()
        res.result_type = self.result_type
        res.path = self.path
        res.timeout = self.timeout
        res.planner_name = self.planner_name
        res.total_exec_time_ms = self.total_exec_time_ms
        res.local_start = self.local_start
        res.local_goal = self.local_goal
        res.goal_direction = self.goal_direction
        res.ego_location = self.ego_location
        res.map_goal = self.map_goal
        res.map_next_goal = self.map_next_goal
        return res
        
