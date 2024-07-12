from model.sensors.odometer import  Odometer
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.camera import Camera


class EgoCar:

    def get_odometer(self) -> Odometer:
        pass

    def get_gps(self) -> GPS:
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
    