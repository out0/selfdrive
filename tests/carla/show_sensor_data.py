import sys
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
from model.discrete_component import DiscreteComponent
import math

class ShowSensorData (DiscreteComponent):
    _ego: CarlaEgoCar
    _gps: CarlaGps
    _imu: CarlaIMU
    _odometer: CarlaOdometer

    def __init__(self, ego: CarlaEgoCar):
        super().__init__(1000)
        self._ego = ego
        self._gps = ego.get_gps()
        self._imu = ego.get_imu()
        self._odometer = ego.get_odometer()

    def _loop(self, dt: float) -> None:
        gps_data: GpsData = self._gps.read()
        print("Gps:")
        print(f"    lat: {gps_data.latitude}")
        print(f"    lon: {gps_data.longitude}")
        
        imu_data = self._imu.read()
        print("IMU")
        print(f"    compass: {math.degrees(imu_data.compass)} degrees")
        print(f"    accel_x = {imu_data.accel_x}")
        print(f"    accel_y = {imu_data.accel_y}")
        print(f"    gyro_x = {imu_data.accel_y}")
        print(f"    gyro_y = {imu_data.gyro_y}")
        
        vel = self._odometer.read()
        print (f"Velocity: {vel} m/s or {3.6*vel} km/h")
        print("")
        print("DEBUG Reference Values")
        print(f"    heading (yaw): {self._ego.get_heading()}")
        print("")
        print(f"Compass heading: {math.degrees(imu_data.compass) - 90}")

client = CarlaClient(town='Town07')
ego = CarlaEgoCar(client)
ego.init_fake_bev_seg_camera()
ego.set_pose(0, 0, 0, 0)
ego.set_autopilot(True)
show_sensor_data = ShowSensorData(ego)
show_sensor_data.start()

input()
show_sensor_data.destroy()
ego.destroy()


#https://python-control.readthedocs.io/en/0.10.0/kincar-fusion.html
#https://digilent.com/blog/how-to-convert-magnetometer-data-into-compass-heading/