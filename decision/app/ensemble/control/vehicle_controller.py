from .. model.planning_data import PlanningData
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planner_executor import LocalPlannerExecutor
from .. model.physical_paramaters import PhysicalParameters
from .. motion.motion_controller import MotionController
from .. planner.interpolator import Interpolator
from .. planner.overtaker import Overtaker
from .. planner.hybrid_a import HybridAStar
from .. planner.bi_rrt import BiRRTStar
from .. planner.ensemble import Ensemble
from pydriveless import EgoVehicle, Camera, SLAM, GPS, IMU, Odometer
from pydriveless import FastStateMachine
from pydriveless import WorldPose, MapPose, Waypoint, angle
from .planning_pipeline import PlanningPipeline
from enum import Enum
import time

#from carladriver import CarlaSimulation

STATE_MACHINE_DEBUG = True
CONTROL_DEBUG = True

MAX_PATH_HOPPING = 5
MOTION_CONTROLLER_LOOP_PERIOD_MS = 5
MOTION_CONTROLLER_LONGITUDINAL_PERIOD_MS = 50
VELOCITY = 1


class VehicleControllerState(Enum):
    CALIBRATE_SLAM = 0
    READY = 1
    DRIVE_PATH_START = 2
    DRIVE_PATH = 3
    INVALID_SELF_POSITIONING = 4
    FINISH_DRIVE = 5
    DRIVE_SEGMENT = 6
    WAIT_FULL_VEHICLE_STOP = 7
    NO_LOCAL_GOAL_FOUND = 8
    LOCAL_PLANNER_MONITORING = 9
    INVALID_GLOBAL_PATH = 10


class LocalPlannerType(Enum):
    INTERPOLATOR = 0
    OVERTAKER = 1
    HYBRID_A = 2
    BI_RRT = 3
    ENSEMBLE = 4


class VehicleController(FastStateMachine):
    __vehicle: EgoVehicle
    __input_camera: Camera
    __slam: SLAM
    __gps: GPS
    __imu: IMU
    __planning_pipeline: PlanningPipeline
    __origin: WorldPose
    __motion_controller: MotionController
    __local_planner_type: LocalPlannerType
    __local_planner_timeout_ms: int
    __local_planner: LocalPlannerExecutor
    __path: list[MapPose]
    __abort: bool
    __last_planning_data: PlanningData

    def __init__(self,
                 vehicle: EgoVehicle,
                 gps: GPS,
                 imu: IMU,
                 input_camera: Camera,
                 odometer: Odometer,
                 slam: SLAM,
                 local_planner_type: LocalPlannerType,
                 local_planner_timeout_ms: int):

        super().__init__(state_map={
            VehicleControllerState.CALIBRATE_SLAM: "_state_calibrate_slam",
            VehicleControllerState.READY:  "_state_ready_wait_global_path",
            VehicleControllerState.DRIVE_PATH_START:  "_state_drive_new_path_start",
            VehicleControllerState.DRIVE_PATH:  "_state_navigate_path",
            VehicleControllerState.INVALID_SELF_POSITIONING:  "_state_invalid_self_positioning",
            VehicleControllerState.FINISH_DRIVE:  "_state_finish_drive",
            VehicleControllerState.DRIVE_SEGMENT:  "_state_drive_segment",
            VehicleControllerState.WAIT_FULL_VEHICLE_STOP:  "_state_wait_full_stop",
            VehicleControllerState.NO_LOCAL_GOAL_FOUND: "_state_no_local_goal_found",
            VehicleControllerState.LOCAL_PLANNER_MONITORING:  "_state_local_planner_monitoring",
            VehicleControllerState.INVALID_GLOBAL_PATH:  "_state_invalid_global_path"
        }, debug=STATE_MACHINE_DEBUG, initial_state=VehicleControllerState.CALIBRATE_SLAM)

        self.__vehicle = vehicle
        self.__input_camera = input_camera
        self.__slam = slam
        self.__gps = gps
        self.__imu = imu
        self.__odometer = odometer
        self.__planning_pipeline = None
        self.__abort = False
        self.__local_planner_type = local_planner_type
        self.__local_planner = None
        self.__local_planner_timeout_ms = local_planner_timeout_ms
        self.__path = None
        self.__last_planning_data = None
        self.__motion_controller = MotionController(
            period_ms=MOTION_CONTROLLER_LOOP_PERIOD_MS,
            longitudinal_controller_period_ms=MOTION_CONTROLLER_LONGITUDINAL_PERIOD_MS,
            ego=self.__vehicle,
            slam=self.__slam,
            odometer=self.__odometer,
            on_finished_motion=self.on_finished_motion
        )

        self.__motion_controller.start()
        self.__motion_controller.brake()



    def on_finished_motion(self, mc) -> None:
        mc.brake()
        pass

    def drive(self, path: list[MapPose]) -> None:
        self.__abort = True
        while not self.is_ready():
            time.sleep(0.01)
        self.__abort = False
        self.__path = path
        self.set_state(VehicleControllerState.DRIVE_PATH_START)

    def is_ready(self) -> bool:
        return self.state()

    def _state_calibrate_slam(self, state: VehicleControllerState) -> VehicleControllerState:
        p = self.__gps
        data = p.read()
        if not data.valid:
            return VehicleControllerState.CALIBRATE_SLAM

        imu_data = self.__imu.read()
        if not imu_data.valid:
            return VehicleControllerState.CALIBRATE_SLAM

        camera_img = self.__input_camera.read()
        if camera_img is None:
            return VehicleControllerState.CALIBRATE_SLAM

        self.__origin = WorldPose(
            lat=angle.new_deg(data.lat),
            lon=angle.new_deg(data.lon),
            alt=data.alt,
            compass=angle.new_deg(imu_data.compass)
        )

        self.__planning_pipeline = PlanningPipeline(self.__origin)
        self.__initialize_local_planner()
        return VehicleControllerState.READY

    def __initialize_local_planner(self):
        match self.__local_planner_type:
            case LocalPlannerType.INTERPOLATOR:
                self.__local_planner = Interpolator(self.__planning_pipeline.get_coord_converter(
                ), max_exec_time_ms=self.__local_planner_timeout_ms)
            case LocalPlannerType.OVERTAKER:
                self.__local_planner = Overtaker(
                    max_exec_time_ms=self.__local_planner_timeout_ms)
            case LocalPlannerType.HYBRID_A:
                self.__local_planner = HybridAStar(
                    max_exec_time_ms=self.__local_planner_timeout_ms)
            case LocalPlannerType.BI_RRT:
                self.__local_planner = BiRRTStar(self.__planning_pipeline.get_coord_converter(),
                                                 max_exec_time_ms=self.__local_planner_timeout_ms,
                                                 max_path_size_px=30,
                                                 dist_to_goal_tolerance_px=5,
                                                 class_cost=PhysicalParameters.SEGMENTATION_CLASS_COST)
            case LocalPlannerType.ENSEMBLE:
                self.__local_planner = Ensemble(self.__planning_pipeline.get_coord_converter(
                ), max_exec_time_ms=self.__local_planner_timeout_ms)

    def _state_ready_wait_global_path(self, state: VehicleControllerState) -> VehicleControllerState:
        if self.__path is None or len(self.__path) <= 0:
            return VehicleControllerState.READY
        return VehicleControllerState.DRIVE_PATH_START

    def _state_drive_new_path_start(self, state: VehicleControllerState) -> VehicleControllerState:
        if CONTROL_DEBUG:
            print(
                f"received a new path to drive with {len(self.__path)} goal points")

        self._path_pos = 0
        self._recovery_from_no_local_goal = False
        return VehicleControllerState.DRIVE_PATH

    def _state_navigate_path(self, state: VehicleControllerState) -> VehicleControllerState:

        self._last_ego_pos = self.__slam.estimate_ego_pose()
        pos = MapPose.find_nearest_goal_pose(
            location=self._last_ego_pos,
            poses=self.__path,
            start=self._path_pos,
            maxHopping=MAX_PATH_HOPPING
        )

        hopping_distance_to_end = (self._path_pos - len(self.__path))

        if pos < 0:
            if hopping_distance_to_end > 1:
                return VehicleControllerState.INVALID_SELF_POSITIONING
            else:
                return VehicleControllerState.FINISH_DRIVE

        if pos == len(self.__path) - 1:
            g1, g2 = self.__path[-1], None
        else:
            g1, g2 = self.__path[pos], self.__path[pos + 1]

        if CONTROL_DEBUG:
            print(
                f"navigating [{pos} --> {pos+1}]. Current pos: {self._last_ego_pos}, g1: {g1}, g2{g2}")

        self._g1 = g1
        self._g2 = g2
        return VehicleControllerState.DRIVE_SEGMENT

    def _state_navig_state_invalid_self_positioningate_path(self, state: VehicleControllerState) -> VehicleControllerState:
        print(
            f"[ERROR] It was not possible to find a valid next goal for the current vehicle position {self._last_ego_pos}.")
        print(f"I'm stopping the vehicle and set it ready to receive a global path again")
        self.__path = None
        self.__motion_controller.brake()
        return VehicleControllerState.WAIT_FULL_VEHICLE_STOP

    def _state_wait_full_stop(self, state: VehicleControllerState) -> VehicleControllerState:
        if self.__odometer.read() > 0.1:
            return VehicleControllerState.WAIT_FULL_VEHICLE_STOP
        return VehicleControllerState.READY

    def _state_finish_drive(self, state: VehicleControllerState) -> VehicleControllerState:
        self.__path = None
        print(f"[SUCCESS] finished driving.")
        return VehicleControllerState.READY

    def __build_local_planning_data(self) -> PlanningData:
        location = self.__slam.estimate_ego_pose()
        frame, ts = self.__input_camera.read()

        planning_data = self.__planning_pipeline.step1_build_planning_data(
            seq=self._path_pos + 1,
            bev=frame,
            ego_location=location,
            g1=self._g1,
            g2=self._g2,
            velocity=VELOCITY
        )

        self.__planning_pipeline.step3_pre_process(planning_data)
        return planning_data

    def _state_drive_segment(self, state: VehicleControllerState) -> VehicleControllerState:
        planning_data = self.__build_local_planning_data()

        if self.__abort:
            self.__abort = False
            return VehicleControllerState.READY

        if not self.__planning_pipeline.step4_find_local_goal(planning_data):
            if self._recovery_from_no_local_goal:
                return VehicleControllerState.INVALID_GLOBAL_PATH

            return VehicleControllerState.NO_LOCAL_GOAL_FOUND

        if self.__abort:
            self.__abort = False
            return VehicleControllerState.READY
        self.__planning_pipeline.step5_perform_local_planning(
            planning_data, self.__local_planner)
        
        self.__last_planning_data = planning_data

        return VehicleControllerState.LOCAL_PLANNER_MONITORING

    def _state_no_local_goal_found(self, state: VehicleControllerState) -> VehicleControllerState:
        # try next pair
        if self.__abort:
            self.__abort = False
            return VehicleControllerState.READY
        if CONTROL_DEBUG:
            print(f"No local goal found, trying again")
        self._path_pos += 1
        self._recovery_from_no_local_goal = True
        return VehicleControllerState.DRIVE_PATH

    def _state_invalid_global_path(self, state: VehicleControllerState) -> VehicleControllerState:
        print(f"[ERROR] unable to plan for pos = {self._path_pos}")
        print(f"I'm stopping the vehicle and set it ready to receive a global path again")
        self.__path = None
        self.__motion_controller.brake()
        return VehicleControllerState.WAIT_FULL_VEHICLE_STOP

    def _state_local_planner_monitoring(self, state: VehicleControllerState) -> VehicleControllerState:
        self._recovery_from_no_local_goal = False
        is_moving = False

        while (self.__local_planner.is_planning() or self.__local_planner.is_optimizing()):
            if self.__abort:
                self.__abort = False
                self.__local_planner.cancel()
                if CONTROL_DEBUG:
                    print(f"Waiting for the LP first response")
                while self.__local_planner.is_running():
                    time.sleep(0.01)
                return VehicleControllerState.READY

            if self.__local_planner.new_path_available():
                is_moving = True
                res = self.__local_planner.get_result()
                map_path = self.__planning_pipeline.step6_translate_local_path_to_map_coordinates(self.__last_planning_data, res)
                if CONTROL_DEBUG:
                    print(
                        f"LP found a new path with {len(res.path)} local points and cost = {res.curve_cost}")
                    print(
                        f"LP path was converted to map path with {len(map_path)} map points")
                    
                #self.sim.show_path(map_path)
                self.__motion_controller.set_path(map_path, self.__last_planning_data.velocity())
                break # TODO: adequar isso

        if self.__local_planner.timeout():
            print("timeout while planning")

        if is_moving:
            while self.__motion_controller.is_moving():
                if self.__abort:
                    self.__abort = False                    
                    self.__motion_controller.cancel()
                    return VehicleControllerState.READY
                time.sleep(0.01)

        return VehicleControllerState.DRIVE_PATH
