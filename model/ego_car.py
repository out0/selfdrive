from model.sensors import Gps, Odometer, IMU
from model.camera import Camera


class EgoCar:

    def get_odometer(self) -> Odometer:
        pass

    def get_gps(self) -> Gps:
        pass

    def get_imu(self) -> IMU:
        pass

    def get_bev_camera(self) -> Camera:
        pass

    def get_front_camera(self) -> Camera:
        pass

    def get_rear_camera(self) -> Camera:
        pass

    def get_left_camera(self) -> Camera:
        pass

    def get_right_camera(self) -> Camera:
        pass
    
    def set_power(self, power: float) -> None:
        pass
    
    def set_brake(self, brake: float) -> None:
        pass
    
    def set_steering(self, angle: int) -> None:
        pass
    