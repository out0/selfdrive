from motion.motion_controller import MotionController
from vision.occupancy_grid_cuda import OccupancyGrid
from enum import Enum
from typing import List
from data.coordinate_converter import CoordinateConverter
from model.discrete_component import DiscreteComponent
from model.map_pose import MapPose
from model.waypoint import Waypoint
from slam.slam import SLAM
from planner.planning_data_builder import PlanningDataBuilder, PlanningData
import math
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType, PlannerResultType, PlanningResult
from planner.collision_detector import CollisionDetector
from model.physical_parameters import PhysicalParameters
from model.ego_car import EgoCar
from slam.slam import SLAM
import time
from utils.telemetry import Telemetry


PROXIMITY_RATE_TO_GET_NEXT_PATH_SEGMENT = 0.9
MOTION_CONTROLLER_EXEC_PERIOD_MS = 5
LONGITUDINAL_EXEC_PERIOD_MS = 100

class ControllerState(Enum):
    ON_HOLD = 0
    START_PLANNING = 1
    WAIT_PLANNING = 2
    EXECUTE_MISSION = 3
    WAIT_MISSION_EXECUTION = 4

class SelfDriveControllerResponseType(Enum):
    UNKNOWN_ERROR = 0
    CANT_LOCATE_IN_GLOBAL_PATH = 1
    PLAN_RETURNED_NONE = 2
    PLAN_INVALID_START = 3
    PLAN_INVALID_GOAL = 4
    PLAN_INVALID_PATH = 5
    MOTION_INVALID_PATH = 6
    GOAL_REACHED = 7
    VALID_WILL_EXECUTE = 8

class SelfDriveControllerResponse:
    response_type: SelfDriveControllerResponseType
    planner_result: PlanningResult
    planner_data: PlanningData
    motion_path: list[MapPose]
    
    def __init__(self,
                response_type: SelfDriveControllerResponseType,
                planner_result: PlanningResult,
                planner_data: PlanningData,
                motion_path: list[MapPose]) -> None:
        self.response_type = response_type
        self.planner_result = planner_result
        self.planner_data = planner_data
        self.motion_path = motion_path
    

class SelfDriveController(DiscreteComponent):
    __local_planner: LocalPlanner
    _motion_controller: MotionController
    _state: ControllerState
    _slam: SLAM
    _planning_data_builder: PlanningDataBuilder
    _on_vehicle_controller_response: callable
    _driving_path: list[MapPose]
    _driving_path_pos: int
    _collision_detector: CollisionDetector
    _last_planning_data: PlanningData
    _coord: CoordinateConverter
    
    
    
    SELF_DRIVE_CONTROLLER_PERIOD_MS = 1
    MOTION_CONTROLLER_PERIOD_MS = 2
    LONGITUDINAL_CONTROLLER_PERIOD_MS = 10
    COLLISION_DETECTOR_PERIOD_MS = 150
    #PLAN_TIMEOUT=1000
    PLAN_TIMEOUT=-1
    SPEED_MS = 2.0


    def __init__(self, 
                 ego: EgoCar, 
                 planning_data_builder: PlanningDataBuilder,
                 slam: SLAM,
                 controller_response: callable,
                 local_planner_type: LocalPlannerType) -> None:
        
        super().__init__(SelfDriveController.SELF_DRIVE_CONTROLLER_PERIOD_MS)
        
        Telemetry.initialize()
        
        self._NAME = "SelfdriveController"
        
        self._slam = slam
        
        first_pose = None
        while first_pose is None:
            first_pose = self._slam.read_gps()
            time.sleep(SelfDriveController.SELF_DRIVE_CONTROLLER_PERIOD_MS/1000)
                
        self._coord = slam.get_coordinate_converter()
        
        self.__local_planner = LocalPlanner(
            plan_timeout_ms=SelfDriveController.PLAN_TIMEOUT,
            local_planner_type=local_planner_type,
            map_coordinate_converter=self._coord
        )
        
        
        self._motion_controller = MotionController(
            period_ms=SelfDriveController.MOTION_CONTROLLER_PERIOD_MS,
            longitudinal_controller_period_ms=SelfDriveController.LONGITUDINAL_CONTROLLER_PERIOD_MS,
            ego=ego,
            slam=self._slam,
            on_finished_motion=self.__on_finished_motion
        )
        self._motion_controller.start()
        
        self._state = ControllerState.ON_HOLD
        
        self._collision_detector = CollisionDetector(
            period_ms=SelfDriveController.COLLISION_DETECTOR_PERIOD_MS,
            coordinate_converter=self._coord,
            planning_data_builder=planning_data_builder,
            on_collision_detected_cb=self.__on_collision_detected
        )
        self._collision_detector.start()
        
        self._on_vehicle_controller_response = controller_response
        self._driving_path = None
        self._driving_path_pos = 0
       
        self._planning_data_builder = planning_data_builder

    
    def destroy(self) -> None:
        self._run = False
        self._collision_detector.destroy()
        self._motion_controller.destroy()
        self.__local_planner.destroy()
        
    def __on_collision_detected(self) -> None:
        msg = f"Collision ahead detected. Performing replan on path pos: {self._driving_path_pos}"
        print(msg)
        self.__replan()
        ## DEBUG!!
        #self._motion_controller.cancel()
        #self._motion_controller.brake()
    
    def __on_finished_motion(self, motion_controller: MotionController) -> None:
        motion_controller.brake()
        print(f"Motion finished succesfuly on path pos: {self._driving_path_pos}")
        self.__replan()
       
    def __replan(self) -> None:
        self._motion_controller.cancel()
        self._motion_controller.brake()
        self._state = ControllerState.START_PLANNING
        
    def cancel(self, brake: bool = True) -> None:
        if brake:
            self._motion_controller.cancel()
            self._motion_controller.brake()

        self.__local_planner.cancel()
        self._state = ControllerState.ON_HOLD
        self._driving_path = None
        self._driving_path_pos = -1
        pass  
    
    def drive(self, path: list[MapPose]) -> None:
        self.cancel()
        self._driving_path = path
        self._state = ControllerState.START_PLANNING

    def _loop(self, dt: float) -> None:
        match self._state:
            case ControllerState.ON_HOLD:
                return
            case ControllerState.START_PLANNING:
                self._state = self.__plan_next()
                return
            case ControllerState.WAIT_PLANNING:
                if not self.__local_planner.is_planning():
                    self._state = ControllerState.EXECUTE_MISSION
                time.sleep(0.001)
                return

            case ControllerState.EXECUTE_MISSION:
                self._state = self.__execute_mission()
                return

            case ControllerState.WAIT_MISSION_EXECUTION:
                time.sleep(0.001)
                return
    
    def __check_plan_data(self, plan_data: PlanningData) -> bool:
        return plan_data is not None and \
                plan_data.og is not None and \
                plan_data.ego_location is not None
    
    def __find_goal(self, plan_data: PlanningData) -> tuple[MapPose, MapPose]:
        pos = MapPose.find_nearest_goal_pose(
            location=plan_data.ego_location,
            poses=self._driving_path,
            start=self._driving_path_pos - 1
        )
        
        pos_from_zero = MapPose.find_nearest_goal_pose(
            location=plan_data.ego_location,
            poses=self._driving_path,
            start=0
        )
        
        print (f"__find_goal: nearest goal: start = {self._driving_path_pos} result: {pos}, result from zero: {pos_from_zero}")
        
        if pos < 0:
            return (None, None)
        
        self._driving_path_pos = pos
        
        if pos < len(self._driving_path) - 1:
            return (self._driving_path[pos], self._driving_path[pos + 1])
        else:
            return (self._driving_path[pos], None)
    
    def __plan_next(self) -> ControllerState:
        plan_data = self._planning_data_builder.build_planning_data()

        if not self.__check_plan_data(plan_data):
            print("Planning data build failed, will wait for valid data")
            return ControllerState.START_PLANNING
        
        p2, p3 = self.__find_goal(plan_data)
        
        if p2 is None:
            print("End of driving mission")
            self.__report_end_of_mission()
            return ControllerState.ON_HOLD
        
        if p3 is None:
            print(f"driving to goal {p2}")
        else:
            print(f"driving to goal {p2}, next goal {p3}")
        
        
        plan_data.set_goals(p2, p3, SelfDriveController.SPEED_MS)
        
        self._last_planning_data = plan_data
        Telemetry.log_pre_planning_data(plan_data)
        
        self.__local_planner.plan(plan_data)
        return ControllerState.WAIT_PLANNING
    
    def __report_end_of_mission(self) -> None:
        self._on_vehicle_controller_response(
            res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.GOAL_REACHED,
                planner_result=None,
                planner_data=self._last_planning_data,
                motion_path=None
            ))
        
    
    def __execute_mission(self) -> ControllerState:
        res = self.__local_planner.get_result()
        
        Telemetry.log_planning_data(self._last_planning_data, res)
        
        match res.result_type:
            case PlannerResultType.NONE:
                self.__on_planning_unknown_error()
                return ControllerState.ON_HOLD
            
            case PlannerResultType.INVALID_START:
                self.__report_invalid_start(res)
                self.cancel()
                return ControllerState.ON_HOLD
            
            case PlannerResultType.INVALID_GOAL:
                self.__report_invalid_goal(res)
                self.cancel()
                return ControllerState.ON_HOLD
            
            case PlannerResultType.INVALID_PATH:
                self.__report_invalid_path(res)
                self.cancel()
                return ControllerState.ON_HOLD
            
            case PlannerResultType.TOO_CLOSE:
                if self._last_planning_data.next_goal is None:
                    self.__report_end_of_mission()
                    return ControllerState.ON_HOLD
                self._driving_path_pos += 1
                return ControllerState.START_PLANNING
        
        ds_path = self.__downsample_waypoints(res.path)
        ideal_motion_path = self._coord.convert_waypoint_path_to_map_pose(res.ego_location, ds_path)
        
        self.__report_executing_mission(res, ideal_motion_path)
        self.__perform_motion(ideal_motion_path)
        return ControllerState.WAIT_MISSION_EXECUTION
    
    def __report_invalid_start(self, res: PlanningResult) -> None:
        self._on_vehicle_controller_response(res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.PLAN_INVALID_START,
                planner_result=res,
                planner_data=self._last_planning_data,
                motion_path=None
            ))
        
    def __report_invalid_goal(self, res: PlanningResult) -> None:
        self._on_vehicle_controller_response(res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.PLAN_INVALID_GOAL,
                planner_result=res,
                planner_data=self._last_planning_data,
                motion_path=None
            ))
    
    def __report_executing_mission(self, res: PlanningResult, path: list[MapPose]) -> None:
        self._on_vehicle_controller_response(res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.VALID_WILL_EXECUTE,
                planner_result=res,
                planner_data=self._last_planning_data,
                motion_path=path
            ))
    
    
    def __report_invalid_path(self, res: PlanningResult) -> None:
        self._on_vehicle_controller_response(res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.PLAN_INVALID_PATH,
                planner_result=res,
                planner_data=self._last_planning_data,
                motion_path=None
            ))
        
    def __on_planning_unknown_error(self) -> None:
        self._on_vehicle_controller_response(res=SelfDriveControllerResponse(
                response_type=SelfDriveControllerResponseType.UNKNOWN_ERROR,
                planner_result=None,
                planner_data=self._last_planning_data,
                motion_path=None
            ))
    

    def __downsample_waypoints(self, waypoints: List[Waypoint]) -> List[Waypoint]:
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

    def __perform_motion(self, path: list[MapPose]) -> None:
        # TODO: must convert path
        self._motion_controller.set_path(path, velocity=4.0)
        self._collision_detector.watch_path(path)