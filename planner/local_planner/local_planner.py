from model.map_pose import MapPose
from model.waypoint import Waypoint
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.goal_point_discover import GoalPointDiscover, GoalPointDiscoverResult
from data.coordinate_converter import CoordinateConverter
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from model.planning_data import *
from enum import Enum
from planner.local_planner.executors.hierarchical_group import HierarchicalGroupPlanner
from planner.local_planner.executors.astar import AStarPlanner
from planner.local_planner.executors.hybridAStar import HybridAStarPlanner
from planner.local_planner.executors.vectorial_astar import VectorialAStarPlanner
from planner.local_planner.executors.overtaker import OvertakerPlanner
from planner.local_planner.executors.interpolator import InterpolatorPlanner
from planner.local_planner.executors.rrt import RRTPlanner


class LocalPlannerType(Enum):
    Ensemble = 0
    Interpolator = 1
    Overtaker = 2
    AStar = 3
    VectorialAStar = 4
    HybridAStar = 5

class LocalPlanner:
    _plan_timeout_ms: int
    _map_coordinate_converter: CoordinateConverter
    _goal_point_discover: GoalPointDiscover
    _path_planner: LocalPathPlannerExecutor
    _planner_result: PlanningResult

    def __init__(self, 
                plan_timeout_ms: int,
                local_planner_type: LocalPlannerType,
                map_coordinate_converter: CoordinateConverter) -> None:
        
        self._plan_timeout_ms = plan_timeout_ms
        self._local_planner_type = local_planner_type
        self._map_coordinate_converter = map_coordinate_converter
        self._goal_point_discover = GoalPointDiscover(map_coordinate_converter)
        self._path_planner = self.__get_local_planner_algorithm(local_planner_type)
        self._planner_result = None
          
    
    def destroy(self) -> None:
        self._path_planner.destroy()
        
    
    def cancel(self):
        self._path_planner.cancel()
        self._planner_result = None

    def is_planning(self) -> bool:
        return self._path_planner.is_planning()
    
    def get_result(self) -> PlanningResult:
        if self._planner_result is not None:
            return self._planner_result
        return self._path_planner.get_result()

    def __get_local_planner_algorithm(self, type: LocalPlannerType) -> LocalPathPlannerExecutor:
        match type:
            case LocalPlannerType.Ensemble:
                return HierarchicalGroupPlanner(
                    self._map_coordinate_converter,
                    self._plan_timeout_ms,
                )
            case LocalPlannerType.Interpolator:
                return InterpolatorPlanner(
                    self._map_coordinate_converter, 
                    self._plan_timeout_ms,
                )
            case LocalPlannerType.Overtaker:
                return OvertakerPlanner(
                    self._plan_timeout_ms,
                    self._map_coordinate_converter,
                )
            case LocalPlannerType.VectorialAStar:
                return VectorialAStarPlanner(
                    self._plan_timeout_ms,
                )
            case LocalPlannerType.AStar:
                return AStarPlanner(
                    self._plan_timeout_ms,
                )
            case LocalPlannerType.HybridAStar:
                return HybridAStarPlanner(
                    self._plan_timeout_ms,
                    self._map_coordinate_converter,
                    10
                )
                
                
    def plan (self, planning_data: PlanningData):
        
        goal_result = self._goal_point_discover.find_goal(og=planning_data.og,
                                                          current_pose=planning_data.ego_location,
                                                          goal_pose=planning_data.goal,
                                                          next_goal_pose=planning_data.next_goal)
    
        if goal_result.goal is None:
            return PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.INVALID_GOAL,
                planning_data, 
                goal_result
            )

        if goal_result.too_close:
            return PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.TOO_CLOSE,
                planning_data, 
                goal_result
            )

        if goal_result.start is None:
            return PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.INVALID_START,
                planning_data, 
                goal_result
            )
        
        planning_data.og.set_goal_vectorized(goal_result.goal)       
                
        self._path_planner.plan(planning_data, goal_result)





    
    

