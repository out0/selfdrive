from model.map_pose import MapPose
from model.waypoint import Waypoint
from vision.occupancy_grid_cuda import OccupancyGrid
from planner.goal_point_discover import GoalPointDiscover, GoalPointDiscoverResult
from data.coordinate_converter import CoordinateConverter
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from model.planning_data import *
from enum import Enum
from planner.local_planner.executors.hierarchical_group import HierarchicalGroupPlanner
# from planner.local_planner.executors.astar import AStarPlanner
from planner.local_planner.executors.hybridAStar import HybridAStarPlanner
# from planner.local_planner.executors.vectorial_astar import VectorialAStarPlanner
from planner.local_planner.executors.overtaker import OvertakerPlanner
from planner.local_planner.executors.interpolator import InterpolatorPlanner
#from planner.local_planner.executors.rrtStar import RRTPlanner
from planner.local_planner.executors.rrtStar2 import RRTPlanner
from planner.local_planner.executors.ensemble import EnsemblePlanner


class LocalPlannerType(Enum):
    Ensemble = 0
    HierarchicalGroup = 1
    Interpolator = 2
    Overtaker = 3
    HybridAStar = 4
    RRTStar = 5
    # AStar = 999
    # VectorialAStar = 999
    

class LocalPlanner:
    __plan_timeout_ms: int
    __map_coordinate_converter: CoordinateConverter
    __goal_point_discover: GoalPointDiscover
    __path_planner: LocalPathPlannerExecutor
    __planner_result: PlanningResult
    __is_verifying_search_data: bool

    def __init__(self, 
                plan_timeout_ms: int,
                local_planner_type: LocalPlannerType,
                map_coordinate_converter: CoordinateConverter) -> None:
        
        self.__plan_timeout_ms = plan_timeout_ms
        self.__map_coordinate_converter = map_coordinate_converter
        self.__goal_point_discover = GoalPointDiscover(map_coordinate_converter)
        self.__path_planner = self.__get_local_planner_algorithm(local_planner_type)
        self.__planner_result = None
        self.__is_verifying_search_data = False
          
    
    def destroy(self) -> None:
        self.__path_planner.destroy()
        
    
    def cancel(self):
        self.__path_planner.cancel()
        self.__planner_result = None
        self.__is_verifying_search_data = False

    def is_planning(self) -> bool:
        return self.__is_verifying_search_data or self.__path_planner.is_planning()
    
    def get_result(self) -> PlanningResult:
        if self.__planner_result is not None:
            return self.__planner_result
        return self.__path_planner.get_result()

    def __get_local_planner_algorithm(self, type: LocalPlannerType) -> LocalPathPlannerExecutor:
        match type:
            case LocalPlannerType.Ensemble:
                return EnsemblePlanner(
                    self.__map_coordinate_converter,
                    self.__plan_timeout_ms,
                )
            case LocalPlannerType.HierarchicalGroup:
                return HierarchicalGroupPlanner(
                    self.__map_coordinate_converter,
                    self.__plan_timeout_ms,
                )
            case LocalPlannerType.Interpolator:
                return InterpolatorPlanner(
                    self.__map_coordinate_converter, 
                    self.__plan_timeout_ms,
                )
            case LocalPlannerType.Overtaker:
                return OvertakerPlanner(
                    self.__plan_timeout_ms,
                    self.__map_coordinate_converter,
                )
            # case LocalPlannerType.VectorialAStar:
            #     return VectorialAStarPlanner(
            #         self.__plan_timeout_ms,
            #     )
            # case LocalPlannerType.AStar:
            #     return AStarPlanner(
            #         self.__plan_timeout_ms,
            #     )
            case LocalPlannerType.HybridAStar:
                return HybridAStarPlanner(
                    self.__plan_timeout_ms,
                    self.__map_coordinate_converter,
                    5.0
                )
            case LocalPlannerType.RRTStar:
                return RRTPlanner(self.__plan_timeout_ms)
                
                
                
    def plan (self, planning_data: PlanningData):
        
        self.__is_verifying_search_data = True
        self.__planner_result = None
        
        goal_result = self.__goal_point_discover.find_goal(og=planning_data.og,
                                                          current_pose=planning_data.ego_location,
                                                          goal_pose=planning_data.goal,
                                                          next_goal_pose=planning_data.next_goal)
    
        if goal_result.goal is None:
            self.__planner_result = PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.INVALID_GOAL,
                planning_data, 
                goal_result
            )
            self.__is_verifying_search_data = False
            return

        if goal_result.too_close:
            self.__planner_result = PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.TOO_CLOSE,
                planning_data, 
                goal_result
            )
            self.__is_verifying_search_data = False
            return

        if goal_result.start is None:
            self.__planner_result = PlanningResult.build_basic_response_data(
                "-",
                PlannerResultType.INVALID_START,
                planning_data, 
                goal_result
            )
            self.__is_verifying_search_data = False
            return
        
        planning_data.og.set_goal_vectorized(goal_result.goal)       
                
        self.__path_planner.plan(planning_data, goal_result)
        self.__is_verifying_search_data = False

    

