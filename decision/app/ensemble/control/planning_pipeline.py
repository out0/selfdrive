from pydriveless import MapPose, Waypoint, WorldPose, SearchFrame, CoordinateConverter
from pygpd import GoalPointDiscover
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planner_executor import LocalPlannerExecutor
import numpy as np
import math
from pydriveless import Telemetry

LOCAL_GOAL_NEAR_THRESHOLD_PX = 250
COMPUTE_EXCLUSION_ZONES = True
TELEMETRY = True  # Set to True to enable telemetry logging

class PlanningPipeline:
    
    __search_frame: SearchFrame
    __coordinate_converter: CoordinateConverter
    __goal_point_discover: GoalPointDiscover
    
    def __init__(self, origin: WorldPose):
        
        self.__search_frame = SearchFrame(
                width=PhysicalParameters.OG_WIDTH,
                height=PhysicalParameters.OG_HEIGHT,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
        
        self.__search_frame.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        self.__search_frame.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
        self.__coordinate_converter = CoordinateConverter(
            origin=origin,
            width=PhysicalParameters.OG_WIDTH,
            height=PhysicalParameters.OG_HEIGHT,
            perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH,
            perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT)
        
        self.__goal_point_discover = GoalPointDiscover(self.__coordinate_converter, proximity_threshold_px=LOCAL_GOAL_NEAR_THRESHOLD_PX)
    
    def get_coord_converter(self) -> CoordinateConverter:
        return self.__coordinate_converter

    def step1_build_planning_data(self, seq: int, ego_location: MapPose, g1: MapPose, g2: MapPose, bev: np.ndarray, velocity: float) -> PlanningData:
        self.__search_frame.set_frame_data(bev)
        
        # TO DO: compute min distance based on the defined velocity
        dist = (PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        
        Telemetry.log_if(TELEMETRY, f"log/sf_bev_{seq}.png", self.__search_frame.get_color_frame())

        return PlanningData(
            seq=seq,
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=velocity,
            og=self.__search_frame,
            min_distance=dist)
        
    def step3_pre_process(self, planning_data: PlanningData) -> None:
        planning_data.og().process_safe_distance_zone(planning_data.min_distance(), compute_vectorized=True)
    
    def step4_find_local_goal(self, planning_data: PlanningData) -> bool:
        Telemetry.log_if(TELEMETRY, f"log/bev_{planning_data.seq}_gpd.txt", planning_data)
        Telemetry.log_if(TELEMETRY, f"log/bev_{planning_data.seq}.png", planning_data.og())

        lp = self.__goal_point_discover.find(
            frame=planning_data.og(),
            ego_pose=planning_data.ego_location(),
            compute_exclusion_zones=COMPUTE_EXCLUSION_ZONES,
            g1=planning_data.g1(),
            g2=planning_data.g2())
        
        planning_data.set_local_goal(lp)
        return lp != None
    
    def step5_perform_local_planning(self, planning_data: PlanningData, executor: LocalPlannerExecutor) -> None:
        planning_data.og().process_distance_to_goal(planning_data.local_goal().x, planning_data.local_goal().z)
        executor.plan(data=planning_data, run_in_main_thread=False)

    def step6_translate_local_path_to_map_coordinates(self, planning_data: PlanningData, res: PlanningResult) -> list[MapPose]:
        if res.result_type != PlannerResultType.VALID: return None
        ds_path = self.__downsample_waypoints(res.path)
        path = self.__coordinate_converter.convert_list_waypoint_to_map(planning_data.ego_location(), ds_path)
        return path
    
    def __downsample_waypoints(self, waypoints: list[Waypoint]) -> list[Waypoint]:
            res = []
            division = max(1, math.floor(len(waypoints) / 20))

            i = 0
            for p in waypoints:
                if i % division == 0:
                    res.append(p)
                i += 1

            if len(waypoints) > 0:
                res.append(waypoints[len(waypoints) - 1])
            return res
