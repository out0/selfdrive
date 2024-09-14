import sys
sys.path.append("../../")
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
import time
from model.map_pose import MapPose
from model.world_pose import WorldPose
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
    _last_gps_data: float
    _last_imu_data: float
    _gps_period_s: float
    _imu_period_s: float
    _gps_calibration_time_s: float
    _gps_is_calibrated: bool
    
    def __init__(self, ego: CarlaEgoCar, log_file: str, gps_period_ms: int, imu_period_ms: int, gps_calibration_time_ms: int ):
        super().__init__(1)
        self.gps = ego.get_gps()
        self.imu = ego.get_imu()
        self.odom = ego.get_odometer()
        self.logs = []
        self.ego = ego
        self.log_file = log_file
        self._last_gps_data = 0
        self._last_imu_data = 0
        self._gps_period_s = gps_period_ms / 1000
        self._imu_period_s = imu_period_ms / 1000
        self._gps_calibration_time_s = gps_calibration_time_ms / 1000
        self._gps_is_calibrated = False
        
    def __log_gps(self) -> None:
        if self.gps.last_read_timestamp() - self._last_gps_data < self._gps_period_s:
            return
        
        self._last_gps_data = self.gps.last_read_timestamp()
            
        imu_data = self.imu.read()
        gps_data = self.gps.read()
        velocity = self.odom.read()
        
        world_pose = WorldPose(
            lat=gps_data.latitude,
            lon=gps_data.longitude,
            alt=gps_data.altitude,
            heading=imu_data.compass
        )
        
        data = {}
        data['gps'] = str(world_pose)
        data['velocity'] = str(velocity)
        data['pose'] = str(self.__read_expected_location())
        data['timestamp'] = self.gps.last_read_timestamp()
        self.logs.append(f"gps#{json.dumps(data)}")
        
    def __read_expected_location(self) -> MapPose:
        location = self.ego.get_location()
        
        return MapPose(x=location[0],
                       y=location[1],
                       z=location[2],
                       heading=self.ego.get_heading())

    def __log_imu(self) -> IMUData:
        
        if self.imu.last_read_timestamp() - self._last_imu_data < self._imu_period_s:
            return

        self._last_imu_data = self.imu.last_read_timestamp()
        imu_data = self.imu.read()
        velocity = self.odom.read()
 
        
        data = {}
        data['imu'] = str(imu_data)
        data['velocity'] = str(velocity)
        data['pose'] = str(self.__read_expected_location())
        data['timestamp'] = self.imu.last_read_timestamp()
        self.logs.append(f"imu#{json.dumps(data)}")
        
    def sample(self, dt: float):
        
        if not self._gps_is_calibrated:
            st = time.time()
            while (time.time() - st) <= self._gps_calibration_time_s:
                self.__log_gps()
                print ("+ calibration gps")
                time.sleep(self._gps_period_s) 
            self._gps_is_calibrated = True
            return
        
        self.__log_imu()        
        self.__log_gps()
        
    def has_calibration_data(self) -> bool:
        return self._gps_is_calibrated
    
    def add_goal_data(self, goal: MapPose, next_goal: MapPose) -> None:
        data = {}
        data['goal'] = str(goal)
        data['next_goal'] = str(next_goal)        
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