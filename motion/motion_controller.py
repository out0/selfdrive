from model.map_pose import MapPose
from model.discrete_component import DiscreteComponent
from motion.longitudinal_controller import LongitudinalController
from motion.lateral_controller import LateralController
from typing import List
from slam.slam import SLAM
from model.sensors.odometer import Odometer
from model.ego_car import EgoCar

class MotionController (DiscreteComponent):
    _longitudinal_controller: LongitudinalController
    _lateral_controller: LateralController
    _on_finished_motion: callable
    _slam: SLAM
    _list: List[MapPose]
    _search_state: bool
    _last_pos: int
    _ego: EgoCar
    _odometer: Odometer
    _desired_speed: float

    MAX_RANGE_SQUARED = 625

    def __init__(self, 
                period_ms: int, 
                longitudinal_controller_period_ms: int,                
                ego: EgoCar,
                slam: SLAM,
                desired_speed: float,
                on_finished_motion: callable) -> None:
        
        super().__init__(period_ms)
        
        self._ego = ego
        self._odometer = ego.get_odometer()

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
        self._on_finished_motion = on_finished_motion
        self._search_state = False
        self._last_pos = 0
        self._desired_speed = desired_speed
        
        # self._longitudinal_controller.start()
    
    def __set_break(self, val: float):
        self._ego.set_brake(val)
        
    def __set_power(self, val: float):
        self._ego.set_power(val)
        
    def __get_velocity(self) -> float:
        return self._odometer.read()
    
    def __set_sterring(self, val: float):
        self._ego.set_steering(val)

    def set_path(self, list: List[MapPose]):
        self._search_state = True
        self._list = list
        self._last_pos = 0
    
    def brake(self) -> None:
        self._lateral_controller.cancel()
        self._longitudinal_controller.set_speed(0.0)
        self._longitudinal_controller.brake(1.0)

    def _loop(self, dt: float) -> None:
        if not self._search_state:
            return
        
        pos = MapPose.find_nearest_goal_pose( self._slam.estimate_ego_pose(), self._list, 0)

        if (pos < 0):
            self._on_finished_motion(self)
            self._search_state = False
            return
    
   
        print (f"[motion] controlling movement from {pos} to {pos+1}")            
        self._last_pos = pos
        if pos >= len(self._list):
            self._on_finished_motion(self)
            self._search_state = False
            return
            
        p1 = self._list[pos - 1]
        p2 = self._list[pos]
        self._lateral_controller.set_reference_path(p1, p2)
        self._longitudinal_controller.set_speed(self._desired_speed)

        self._lateral_controller.loop(dt)
        self._longitudinal_controller.loop(dt)
    
    def get_driving_path_ahead (self) -> list[MapPose]:
        if not self._search_state:
            return None
        
        pos = MapPose.find_nearest_goal_pose( self._slam.estimate_ego_pose(), self._list, self._last_pos)
        if pos < 0:
            return None
        
        return self._list[pos:]
    
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