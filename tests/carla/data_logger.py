import sys
sys.path.append("../../")
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
import time
from model.map_pose import MapPose
from model.sensor_data import *
from model.discrete_component import DiscreteComponent
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
import json

class DataLogger(DiscreteComponent):
    ego: CarlaEgoCar
    gps: GPS
    imu: IMU
    odom: Odometer
    logs: list[str]
    log_file: str
    
    def __init__(self, sample_time_ms: int, ego: CarlaEgoCar, log_file: str):
        super().__init__(sample_time_ms)
        self.gps = ego.get_gps()
        self.imu = ego.get_imu()
        self.odom = ego.get_odometer()
        self.logs = []
        self.ego = ego
        self.log_file = log_file
        
    def sample(self, dt: float):
        gps_data = self.gps.read()
        imu_data = self.imu.read()
        velocity = self.odom.read()
        location = self.ego.get_location()
        
        pose = MapPose(x=location[0],
                       y=location[1],
                       z=location[2],
                       heading=self.ego.get_heading())
        
        data = {}
        data['gps'] = str(gps_data)
        data['imu'] = str(imu_data)
        data['velocity'] = str(velocity)
        data['pose'] = str(pose)
        data['timestamp'] = str(dt)
        
        self.logs.append(json.dumps(data))
    
    def _loop (self, dt: float):
        self.sample(dt)
        pass


    def destroy(self) -> None:
        super().destroy()
        time.sleep(0.1)
        
        with open(self.log_file, "w") as f:
            for l in self.logs:
                f.write(l + "\n")