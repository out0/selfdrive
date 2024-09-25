from model.map_pose import MapPose
from model.waypoint import Waypoint
import numpy as np
from enum import Enum
from io import StringIO
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
import json


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
    __bev: np.ndarray
    __og: OccupancyGrid
    __ego_location: MapPose
    __velocity: float
    __goal: MapPose
    __next_goal: MapPose

    def __init__(self, bev: np.ndarray, ego_location: MapPose, velocity: float, goal: MapPose, next_goal: MapPose) -> None:
        self.__bev = bev
        
        if bev is not None:     
            self.__og = OccupancyGrid(
                frame=bev,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
        else:
            self.__og = None
            
        self.__ego_location = ego_location
        self.__velocity = velocity
        self.__goal = goal
        self.__next_goal = next_goal
    
    @property
    def bev(self) -> np.ndarray:
        return self.__bev
    
    @property
    def og(self) -> np.ndarray:
        return self.__og
    
    @property
    def ego_location(self) -> np.ndarray:
        return self.__ego_location
    
    @property
    def velocity(self) -> np.ndarray:
        return self.__velocity
    
    @property
    def goal(self) -> np.ndarray:
        return self.__goal
    
    @property
    def next_goal(self) -> np.ndarray:
        return self.__next_goal
   
    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"ego_location:{self.__ego_location},velocity:{self.__velocity},bev:{self.__frame_shape_to_str(self.__bev)},goal:{self.__goal},next_goal:{self.__next_goal}"

    def set_goals(self, goal: MapPose, next_goal: MapPose, expected_velocity: float) -> None:
        self.__goal = goal
        self.__next_goal = next_goal
        self.__velocity = expected_velocity

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

    # def __str__(self) -> str:
    #     str = StringIO()

    #     if self.result_type == PlannerResultType.NONE:
    #         return "-"
    #     elif self.result_type == PlannerResultType.VALID:
    #         return f"({self.ego_location} -> {self.map_goal}) valid plan to goal, waypoint: {self.local_goal}, timeout: {self.timeout}"
    #     elif self.result_type == PlannerResultType.INVALID_START:
    #         return f"({self.ego_location} -> {self.map_goal}) INVALID **start**, waypoint: {self.local_goal}, timeout: {self.timeout}"
    #     elif self.result_type == PlannerResultType.INVALID_PATH:
    #         return f"({self.ego_location} -> {self.map_goal}) INVALID **plan**, waypoint: {self.local_goal}, timeout: {self.timeout}"
    #     elif self.result_type == PlannerResultType.INVALID_GOAL:
    #         return f"({self.ego_location} -> {self.map_goal}) INVALID **goal**, waypoint: {self.local_goal}, timeout: {self.timeout}"
    #     return str.getvalue()
    
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


        

    def __str__(self) -> str:
        path = []
        if self.path is not None:
            for p in self.path:
                path.append(str(p))
        
        data = {
            'result_type': int(self.result_type.value),
            'path': path,
            'timeout': self.timeout,
            'planner_name': self.planner_name,
            'total_exec_time_ms': self.total_exec_time_ms,
            'local_start': str(self.local_start),
            'local_goal': str(self.local_goal),
            'goal_direction': self.goal_direction,
            'ego_location': str(self.ego_location),
            'map_goal': str(self.map_goal),
            'map_next_goal': str(self.map_next_goal)
        }
        return json.dumps(data)
    
    @classmethod
    def from_str(cls, val: str) -> 'PlanningResult':
        res = PlanningResult()
        
        data = json.loads(val)
        
        res.result_type = PlannerResultType(int(data['result_type']))
        
        res.path = []        
        for str_p in data['path']:
            res.path.append(
                Waypoint.from_str(str_p)
            )
        
        res.timeout = bool(data['timeout'])
        res.planner_name = data['planner_name']
        res.total_exec_time_ms = float(data['total_exec_time_ms'])
        res.local_start = Waypoint.from_str(data['local_start'])
        res.local_goal = Waypoint.from_str(data['local_goal'])
        res.goal_direction = int(data['goal_direction'])
        res.ego_location = MapPose.from_str(data['ego_location'])
        res.map_goal = MapPose.from_str(data['map_goal'])
        res.map_next_goal = MapPose.from_str(data['map_next_goal'])        
        return res
        