from carlasim.carla_client import CarlaClient
import carla
import threading
import math, numpy as np
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import *


class PeriodicDataSensor:
    _sensor: any
    _last_sensor_data: any
    _lock: threading.Lock    
    
    def __init__(self, bp: str, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        sensor_bp = client.get_blueprint(bp)
        sensor_bp.set_attribute('sensor_tick', str(capture_period_in_seconds))
        location = carla.Location(x=0.0, y=0.0, z=1)
        rotation = carla.Rotation()
        transform = carla.Transform(location, rotation)
        self._sensor = client.get_world().spawn_actor(sensor_bp, transform, attach_to=vehicle)
        self._sensor.listen(self.__new_data)
        self._lock = threading.Lock()
        self._last_sensor_data = None

    def destroy(self) -> None:
        if self._sensor is None:
            return
        self._sensor.destroy()
        self._sensor = None
        
    def __new_data(self, sensor_data: any):
        if self._lock.acquire(blocking=False):
            self._last_sensor_data = sensor_data
            self._lock.release()
    
    def read(self) -> any:
        while self._sensor is not None:
            if self._lock.acquire(blocking=True, timeout=0.5):
                f = self._last_sensor_data
                self._lock.release()
                return f
        return None



class CarlaGps (PeriodicDataSensor, GPS):    
    def __init__(self, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        super().__init__("sensor.other.gnss", client, vehicle, capture_period_in_seconds)

    def get_location(self) -> np.array:
        p = self._sensor.get_location()
        return np.array([p.x, p.y, p.z])

    def read(self) -> GpsData:
        carla_gps = super().read()
        if carla_gps is None:
            return None
        return GpsData(carla_gps.latitude, carla_gps.longitude, carla_gps.altitude)
        
class CarlaIMU (PeriodicDataSensor, IMU):    
    def __init__(self, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        super().__init__("sensor.other.imu", client, vehicle, capture_period_in_seconds)

    def read(self) -> IMUData:
        carla_imu = super().read()
        if carla_imu is None:
            return None
        data = IMUData()
        data.accel_x = carla_imu.accelerometer.x
        data.accel_y = carla_imu.accelerometer.y
        data.accel_z = carla_imu.accelerometer.z
        data.compass = carla_imu.compass
        data.gyro_x = carla_imu.gyroscope.x
        data.gyro_y = carla_imu.gyroscope.y
        data.gyro_z = carla_imu.gyroscope.z        
        return data

class OdometerData:
    vel_x: float
    vel_y: float
    vel_z: float
    
    def __init__(self) -> None:
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0


class CarlaOdometer (Odometer):
    
    _vehicle: any
    
    def __init__(self, vehicle: any) -> None:
        self._vehicle = vehicle


    def read(self) -> float:
        velocity = self._vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)
    
    def destroy(self) -> None:
        pass