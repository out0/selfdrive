
from model.sensors.odometer import Odometer
from model.sensors.imu import IMU
from model.sensors.gps import GPS
from model.sensor_data import IMUData, GpsData



class Gps:
    def read(self) -> GpsData:
        pass

class Odometer:
    def read(self) -> float:
        pass

class IMU:
    def read(self) -> IMUData:
        pass