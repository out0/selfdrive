from model.vehicle_pose import VehiclePose
from model.discrete_component import DiscreteComponent
from motion.longitudinal_controller import LongitudinalController
from motion.lateral_controller import LateralController
from model.reference_path import ReferencePath
from typing import List
from model.slam import SLAM
from model.sensors import Odometer
from utils.debug import DebugTelemetry

class MotionController (DiscreteComponent):
    _longitudinal_controller: LongitudinalController
    _lateral_controller: LateralController
    _on_finished_motion: callable
    _slam: SLAM
    _list: List[VehiclePose]
    _search_state: bool
    _last_pos: int

    MAX_RANGE_SQUARED = 625

    def __init__(self, 
                period_ms: int, 
                longitudinal_controller_period_ms: int,
                on_finished_motion: callable,
                odometer: callable, 
                power_actuator: callable, 
                brake_actuator: callable,
                steering_actuator: callable,
                slam: SLAM) -> None:
        
        super().__init__(period_ms)

        self._longitudinal_controller = LongitudinalController(
            longitudinal_controller_period_ms,
            brake_actuator=brake_actuator,
            power_actuator=power_actuator,
            odometer=odometer
        )
        self._lateral_controller = LateralController(
            vehicle_length=2,
            odometer=odometer,
            steering_actuator=steering_actuator,
            slam=slam
        )
        self._odometer = odometer
        self._slam = slam
        self._on_finished_motion = on_finished_motion
        self._search_state = False
        self._last_pos = 0
        
        # self._longitudinal_controller.start()

    def set_path(self, list: List[VehiclePose]):
        self._search_state = True
        self._list = list
        self._last_pos = 0
        DebugTelemetry.log_path("[motion controller]", list)

    
    def brake(self) -> None:
        self._lateral_controller.cancel()
        self._longitudinal_controller.set_speed(0.0)
        self._longitudinal_controller.brake(1.0)

    def _loop(self, dt: float) -> None:
        if not self._search_state:
            return
        
        pos = ReferencePath.find_best_p1_for_location(self._list, self._slam.estimate_ego_pose(), self._last_pos)
    
        if (pos < 0):
            DebugTelemetry.log_message("[motion controller] finished motion")
            self._on_finished_motion()
            self._search_state = False
            return
    
   
        #print (f"[motion] controlling movement from {pos} to {pos+1}")            
        self._last_pos = pos
        if pos >= len(self._list) - 1:
            self._on_finished_motion()
            self._search_state = False
            return
            
        p1 = self._list[pos]
        p2 = self._list[pos + 1]
        self._lateral_controller.set_reference_path(p1, p2)
        self._longitudinal_controller.set_speed(p1.desired_speed)

        self._lateral_controller.loop(dt)
        self._longitudinal_controller.loop(dt)
    
    def get_driving_path_ahead (self) -> list[VehiclePose]:
        if not self._search_state:
            return None
        
        pos = ReferencePath.find_best_p1_for_location(self._list, self._slam.estimate_ego_pose(), self._last_pos)
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