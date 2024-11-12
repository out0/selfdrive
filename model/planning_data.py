from model.map_pose import MapPose
from model.waypoint import Waypoint
import numpy as np
from enum import Enum
from io import StringIO
from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
import json
from planner.goal_point_discover import GoalPointDiscoverResult
from typing import Union

def nullable_clone(p: Union[Waypoint, MapPose]) -> Waypoint:
    if p is None: return None
    return p.clone()


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
    __unseg_bev: np.ndarray
    __bev: np.ndarray
    __og: OccupancyGrid
    __ego_location: MapPose
    __ego_location_ubev: MapPose
    __velocity: float
    __goal: MapPose
    __next_goal: MapPose

    def __init__(self, unseg_bev: np.ndarray, bev: np.ndarray, ego_location: MapPose, velocity: float, goal: MapPose, next_goal: MapPose, ego_location_ubev: MapPose) -> None:
        self.__bev = bev
        self.__unseg_bev = unseg_bev
        self.__ego_location_ubev = ego_location_ubev
        
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
    def unseg_bev(self) -> np.ndarray:
        return self.__unseg_bev
    
    @property
    def og(self) -> OccupancyGrid:
        return self.__og
    
    @property
    def ego_location(self) -> MapPose:
        return self.__ego_location
    
    @property
    def ego_location_ubev(self) -> MapPose:
        return self.__ego_location_ubev
        
    @property
    def velocity(self) -> float:
        return self.__velocity
    
    @property
    def goal(self) -> MapPose:
        return self.__goal
    
    @property
    def next_goal(self) -> MapPose:
        return self.__next_goal
   
    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"ego_location:{self.__ego_location},velocity:{self.__velocity},bev:{self.__frame_shape_to_str(self.__bev)},goal:{self.__goal},next_goal:{self.__next_goal},ego_location_ubev:{self.__ego_location_ubev}"

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
    __result_type: PlannerResultType
    __path: list[Waypoint]
    __timeout: bool
    __planner_name: str
    __total_exec_time_ms: int
    __local_start: Waypoint
    __local_goal: Waypoint
    __goal_direction: int
    __ego_location: MapPose
    __map_goal: MapPose
    __map_next_goal: MapPose
    
    def __init__(self, 
                 planner_name: str,
                 ego_location: MapPose,
                 goal: MapPose,
                 next_goal: MapPose,
                 local_start: Waypoint,
                 local_goal: Waypoint,
                 direction: int,
                 timeout: bool,
                 path: list[Waypoint],
                 result_type: PlannerResultType,
                 total_exec_time_ms: float
                 ) -> None:
        self.__result_type = result_type
        self.__path = path
        self.__timeout = timeout
        self.__planner_name = planner_name
        self.__total_exec_time_ms = 0
        self.__local_start = nullable_clone(local_start)
        self.__local_goal  = nullable_clone(local_goal)
        self.__goal_direction = direction
        self.__ego_location = nullable_clone(ego_location)
        self.__map_goal = nullable_clone(goal)
        self.__map_next_goal = nullable_clone(next_goal)
        self.__total_exec_time_ms = total_exec_time_ms

    @property
    def result_type (self) -> PlannerResultType:
        return self.__result_type
    
    @property
    def path (self) -> list[Waypoint]:
        return self.__path
    
    @property
    def timeout (self) -> bool:
        return self.__timeout
    
    @property
    def planner_name (self) -> str:
        return self.__planner_name
    
    @property
    def total_exec_time_ms (self) -> int:
        return self.__total_exec_time_ms
    
    @property
    def local_start (self) -> Waypoint:
        return self.__local_start
    
    @property
    def local_goal (self) -> Waypoint:
        return self.__local_goal

    @property
    def goal_direction (self) -> int:
        return self.__goal_direction

    @property
    def ego_location (self) -> MapPose:
        return self.__ego_location

    @property
    def map_goal (self) -> MapPose:
        return self.__map_goal

    @property
    def map_next_goal (self) -> MapPose:
        return self.__map_next_goal
    
    @property
    def total_exec_time_ms(self) -> float:
        return self.__total_exec_time_ms

    def update_path(self, smooth_path: list[Waypoint]) -> None:
        self.path = smooth_path
    
    def update_planner_name(self, name: str) -> None:
        self.planner_name = name


    def __str__(self) -> str:
        path = []
        if self.__path is not None:
            for p in self.__path:
                path.append(str(p))
        
        data = {
            'result_type': int(self.__result_type.value),
            'path': path,
            'timeout': self.__timeout,
            'planner_name': self.__planner_name,
            'total_exec_time_ms': self.__total_exec_time_ms,
            'local_start': str(self.__local_start),
            'local_goal': str(self.__local_goal),
            'goal_direction': self.__goal_direction,
            'ego_location': str(self.__ego_location),
            'map_goal': str(self.__map_goal),
            'map_next_goal': str(self.__map_next_goal)
        }
        return json.dumps(data)
    
    @classmethod
    def from_str(cls, val: str) -> 'PlanningResult':       
        data = json.loads(val)
        
        result_type = PlannerResultType(int(data['result_type']))
        
        path = []        
        for str_p in data['path']:
            path.append(
                Waypoint.from_str(str_p)
            )
        
        timeout = bool(data['timeout'])
        planner_name = data['planner_name']
        total_exec_time_ms = float(data['total_exec_time_ms'])
        local_start = Waypoint.from_str(data['local_start'])
        local_goal = Waypoint.from_str(data['local_goal'])
        goal_direction = int(data['goal_direction'])
        ego_location = MapPose.from_str(data['ego_location'])
        map_goal = MapPose.from_str(data['map_goal'])
        map_next_goal = MapPose.from_str(data['map_next_goal'])
        
        return PlanningResult(
            result_type=result_type,
            path = path,
            timeout = timeout,
            planner_name = planner_name,
            total_exec_time_ms = total_exec_time_ms,
            local_start = local_start,
            local_goal = local_goal,
            direction= goal_direction,
            ego_location = ego_location,
            goal = map_goal,
            next_goal = map_next_goal
        )
    
    @classmethod
    def build_basic_response_data(cls, planner_name: str, result_type: PlannerResultType, planning_data: PlanningData, goal_result: GoalPointDiscoverResult, total_exec_time_ms: int = 0) -> 'PlanningResult':
        return  PlanningResult(
            planner_name = planner_name,
            ego_location = nullable_clone(planning_data.ego_location),
            goal = nullable_clone(planning_data.goal),
            next_goal = nullable_clone(planning_data.next_goal),
            local_start = nullable_clone(goal_result.start),
            local_goal = nullable_clone(goal_result.goal),
            direction = goal_result.direction,
            timeout = 0,
            path = None,
            result_type = result_type,
            total_exec_time_ms=total_exec_time_ms
        )
        
class CollisionReport:
    __primitives: list[Waypoint]
    __watch_path: list[MapPose]
    __collision_time: float
    __watch_target: MapPose
    __ego_location: MapPose
    
    def __init__(self, 
                 primitives: list[Waypoint],
                 watch_path: list[MapPose],
                 watch_target: MapPose,
                 ego_location: MapPose,
                 collision_time: float):
        self.__primitives = primitives
        self.__watch_path = watch_path
        self.__collision_time = collision_time
        self.__watch_target = watch_target
        self.__ego_location = ego_location
        
    @property
    def primitives (self) -> list[Waypoint]:
        return self.__primitives
    
    @property
    def watch_path (self) -> list[MapPose]:
        return self.__watch_path
    
    @property
    def collision_time (self) -> float:
        return self.__collision_time    
    
    @property
    def watch_target (self) -> MapPose:
        return self.__watch_target

    @property
    def ego_location(self) -> MapPose:
        return self.__ego_location

    def __str__(self) -> str:
        str_primitives = []
        if self.__primitives is not None:
            for p in self.__primitives:
                str_primitives.append(str(p))

        str_watch_path = []
        if self.__watch_path is not None:
            for p in self.__watch_path:
                str_watch_path.append(str(p))

        data = {
            'primitives': str_primitives,
            'watch_path': str_watch_path,
            'collision_time': self.__collision_time,
            'watch_target': str(self.__watch_target),
            'ego_location': str(self.__ego_location)
        }
        return json.dumps(data)
    
    @classmethod
    def from_str(cls, val: str) -> 'PlanningResult':       
        data = json.loads(val)
        
        primitives = []
        for str_p in data['primitives']:
            primitives.append(
                Waypoint.from_str(str_p)
            )

        watch_path = []
        for str_p in data['watch_path']:
            watch_path.append(
                MapPose.from_str(str_p)
            )
        
        return CollisionReport(
            primitives=primitives,
            watch_path=watch_path,
            watch_target=MapPose.from_str(data['watch_target']),
            collision_time=float(data['collision_time']),
            ego_location=MapPose.from_str(data['ego_location'])
        )