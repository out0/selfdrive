from pydriveless import MapPose, Odometer
from pydriveless import DiscreteComponent, EgoVehicle
from .longitudinal_controller import LongitudinalController
from .lateral_controller import LateralController
from typing import List
from pydriveless import SLAM
from pydriveless import Telemetry
import json

SHOW_DEBUG_MESSAGES = True
TELEMETRY = False

class MotionController (DiscreteComponent):
    _longitudinal_controller: LongitudinalController
    _lateral_controller: LateralController
    _slam: SLAM
    _path: List[MapPose]
    _search_state: bool
    _last_pos: int
    _ego: EgoVehicle
    _desired_speed: float

    MAX_RANGE_SQUARED = 625

    def __init__(self, 
                period_ms: int, 
                longitudinal_controller_period_ms: int,                
                ego: EgoVehicle,
                slam: SLAM,
                odometer: Odometer) -> None:
        
        super().__init__(period_ms)
        
        self._ego = ego
        self._odometer = odometer

        self._longitudinal_controller = LongitudinalController(
            longitudinal_controller_period_ms,
            brake_actuator=self.__set_break,
            power_actuator=self.__set_power,
            velocity_read=self.__get_velocity
        )
        self._lateral_controller = LateralController(
            vehicle_length=2,
            velocity_read=self.__get_velocity,
            steering_actuator=self.__set_sterring,
            slam=slam
        )
        self._slam = slam
        self._search_state = False
        self._last_pos = 0
        self._desired_speed = 0

        
        # self._longitudinal_controller.start()
    
    def __set_break(self, val: float):
        print (f"braking with {val}")
        self._ego.set_brake(val)
        
    def __set_power(self, val: float):
        print (f"set power {val}")
        self._ego.set_power(val)
        
    def __get_velocity(self) -> float:
        return self._odometer.read()
    
    def __set_sterring(self, val: float):
        self._ego.set_steering(val)

    def set_path(self, path: List[MapPose], velocity: float):
        self._search_state = True
        self._path = MapPose.remove_repeated_seq_points_in_list(path)
        self._last_pos = 0
        self._desired_speed = velocity
    
    def brake(self) -> None:
        self._lateral_controller.cancel()
        self._longitudinal_controller.brake(1.0)
    
    def is_tracking(self) -> bool:
        return self._search_state

    def _loop(self, dt: float) -> None:
        if not self._search_state:
            return
        
        pose = self._slam.estimate_ego_pose()
        
        pos = MapPose.find_nearest_goal_pose( pose , self._path, 0, max_hopping=len(self._path) - 1)
        
        if TELEMETRY:
            data = {
                "pose" : str(pose),
                "pos": pos
            }
            Telemetry.log("log/motion_controller.log", json.dumps(data), append=True)

        if (pos < 0):
            self._search_state = False
            return
    
        if SHOW_DEBUG_MESSAGES:
            print (f"[motion] controlling movement from {pos} to {pos+1}")            
        
        self._last_pos = pos
        if pos >= len(self._path) - 1:
            self._search_state = False
            return
            
        p1 = self._path[pos - 1]
        p2 = self._path[pos]
        self._lateral_controller.set_reference_path(p1, p2)
        self._longitudinal_controller.set_speed(self._desired_speed)

        self._lateral_controller.loop(dt)
        self._longitudinal_controller.loop(dt)
    
    def get_path_pos(self) -> int:
        return self._last_pos

    def get_driving_path_ahead (self) -> list[MapPose]:
        if not self._search_state:
            return None
        
        pos = MapPose.find_nearest_goal_pose( self._slam.estimate_ego_pose(), self._path, self._last_pos)
        if pos < 0:
            return None
        
        return self._path[pos:]
    
    def get_crosstrack_error(self) -> float:
        return self._lateral_controller.get_crosstrack_error()

    def get_heading_error(self) -> float:
        return self._lateral_controller.get_heading_error()
    
    def cancel(self) -> None:
        self._search_state = False
        self.brake()

    def destroy(self) -> None:
        self.cancel()
        super().destroy()