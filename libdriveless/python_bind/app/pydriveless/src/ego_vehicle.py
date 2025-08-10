from .sensors.gps import GpsData
from .sensors.imu import IMUData


class EgoVehicle:
    
    def __init__(self):
        pass
    
    def set_power(self, power_level: float) -> None:
        pass
    
    def set_brake(self, brake_level: float) -> None:
        pass
    
    def set_steering(self, angle: float) -> None:
        pass
    
    def read_gps(self) -> GpsData:
        pass
    
    def read_imu(self) -> IMUData:
        pass
    
    