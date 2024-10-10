from model.waypoint import Waypoint
from model.world_pose import WorldPose
from model.map_pose import MapPose
from slam.slam import SLAM
import math

class LateralController:
    __K_GAIN: float = 3
    __V_DUMPING_CONSTANT: float = 2
    __MAX_RANGE: float = 40.0
    _slam: SLAM
    _velocity_read: callable
    _steering_actuator: callable
    _vehicle_length: float
    _p1: MapPose
    _p2: MapPose
    _canceled: bool
    
    _last_crosstrack_error: float
    _last_heading_error: float

    def __init__(self, vehicle_length: float, slam: SLAM, velocity_read: callable, steering_actuator: callable) -> None:
        self._slam = slam
        self._velocity_read = velocity_read
        self._steering_actuator = steering_actuator
        self._vehicle_length = vehicle_length
        self._canceled = True
        self._last_crosstrack_error = 0
        self._last_heading_error = 0

    def set_reference_path(self, p1: MapPose, p2: MapPose):
        self._p1 = p1
        self._p2 = p2
        self._canceled = False

    def __get_ref_point(self) -> MapPose:
        cg: MapPose = self._slam.estimate_ego_pose()
        a = math.radians(cg.heading)
        return MapPose(x=cg.x + math.cos(a) * self._vehicle_length, y=cg.y + math.sin(a) * self._vehicle_length, z=0, heading=cg.heading)

    def __fix_range(heading: float) -> float:
        return min(
                max(heading, -LateralController.__MAX_RANGE),
                LateralController.__MAX_RANGE)
    
    def cancel(self):
        self._canceled = True

    def get_crosstrack_error(self) -> float:
        if self._canceled:
            return 0
        
        return self._last_crosstrack_error

    def get_heading_error(self) -> float:
        if self._canceled:
            return 0
        
        return self._last_heading_error

    
    def loop(self, dt: float) -> None:
        if self._canceled:
            return
        
        ego_ref = self.__get_ref_point()

        current_speed = self._velocity_read()
        if current_speed < 0.1:
            return

        path_heading = MapPose.compute_path_heading(self._p1,  self._p2)
        crosstrack_error = MapPose.distance_to_line(self._p1, self._p2, ego_ref)
        heading_error = path_heading - math.radians(ego_ref.heading)
        

        self._last_crosstrack_error = crosstrack_error
        self._last_heading_error = heading_error

        if current_speed > 0:
            new_heading = math.degrees(heading_error + math.atan( LateralController.__K_GAIN * crosstrack_error / (current_speed + LateralController.__V_DUMPING_CONSTANT)))
            new_heading = LateralController.__fix_range(new_heading)
            self._steering_actuator(new_heading)