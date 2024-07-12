import carla
from carlasim.carla_ego_car import EgoCar
from carlasim.carla_client import CarlaClient
from model.map_pose import VehiclePose
from model.discrete_component import DiscreteComponent
from slam.slam import SLAM
from utils.datalink.pybind.pydatalink import PyDatalink
from utils.float_encoder import FloatEncoder
import time
import numpy as np
from typing import List
import math

PRE_PLANNING_DATA_PORT = 19990
SENSOR_PORT = 19991
CAR_CONTROL_PORT = 19992
CAR_DEBUG_PORT = 19993


class PrePlanningDataSender(DiscreteComponent):
    _ego_car: EgoCar
    _datalink: PyDatalink
    _send_as_bev: bool

    def __init__(self, period_ms: int, ego: EgoCar, send_as_bev: bool) -> None:
        super().__init__(period_ms)
        self._ego_car = ego
        self._send_as_bev = send_as_bev
        self._frames_size = -1
        self._datalink = PyDatalink(host=None, port=PRE_PLANNING_DATA_PORT, timeout_ms=1000)
    
    def wait_ready(self) -> None:
        while not self._datalink.is_connected():
            time.sleep(0.05)

    def __append_float(arr: np.ndarray, row: int, val: float) -> int:
        valb = FloatEncoder.encode(val)
        c = 0
        for b in valb:
            arr[row, c] = b
            c += 1
        return c
    
    def __get_frame_size(frame: np.ndarray) -> int:
        i = 1
        for l in frame.shape:
            i *= l
        return i

    def __sum_frames_size(frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray, frame4: np.ndarray):
        total = 0
        total += PrePlanningDataSender.__get_frame_size(frame1)
        total += PrePlanningDataSender.__get_frame_size(frame2)
        total += PrePlanningDataSender.__get_frame_size(frame3)
        total += PrePlanningDataSender.__get_frame_size(frame4)
        return total

    def _loop(self, dt: float) -> None:
        if not self._datalink.is_connected():
            return

        frame = self._ego_car.get_bev_camera().read()
        if frame is None:
            return

        pose = self._ego_car.get_slam().estimate_ego_pose()
        velocity = self._ego_car.get_odometer().read()
        gps = self._ego_car.get_gps().read()
        
        if self._send_as_bev:
            frame1 = self._ego_car.get_bev_camera().read()
        else:
            frame1 = self._ego_car.get_front_camera().read()
            frame2 = self._ego_car.get_left_camera().read()
            frame3 = self._ego_car.get_right_camera().read()
            frame4 = self._ego_car.get_rear_camera().read()

        sensor_data = np.zeros((10, 8), dtype=np.uint8)
        PrePlanningDataSender.__append_float(sensor_data, 0, pose.x)
        PrePlanningDataSender.__append_float(sensor_data, 1, pose.y)
        PrePlanningDataSender.__append_float(sensor_data, 2, pose.heading)
        PrePlanningDataSender.__append_float(sensor_data, 3, velocity)
        PrePlanningDataSender.__append_float(sensor_data, 4, gps.latitude)
        PrePlanningDataSender.__append_float(sensor_data, 5, gps.longitude)
        PrePlanningDataSender.__append_float(sensor_data, 6, gps.altitude)
        
        if self._frames_size < 0:
            if self._send_as_bev:
                self._frames_size = PrePlanningDataSender.__get_frame_size(frame1)
            else:
                self._frames_size = PrePlanningDataSender.__sum_frames_size(frame1, frame2, frame3, frame4)
        
            self._sensor_data_size = PrePlanningDataSender.__get_frame_size(sensor_data)
        
        if self._send_as_bev:
            frames_data = frame1.reshape((self._frames_size))
        else:
            frames_data = np.stack((frame1, frame2, frame3, frame4), axis=0).reshape((self._frames_size))
        
        data_to_send = np.concatenate((sensor_data.reshape((self._sensor_data_size)), frames_data))
        self._datalink.write_uint8_np(data_to_send)


class RemoteSensorsServer(DiscreteComponent):   
    _sensor_link: PyDatalink
    _slam: SLAM
    
    def __init__(self,  period_ms: int, ego: EgoCar, slam: SLAM) -> None:
        super().__init__(period_ms)
        self._sensor_link = PyDatalink(host=None, port=SENSOR_PORT, timeout_ms=1000)
        self._ego = ego
        self._slam = slam
    
    def wait_ready(self) -> None:
        while not self._sensor_link.is_connected():
            time.sleep(0.05)

    def _loop(self, dt: float) -> None:
        if not self._sensor_link.is_connected():
            return

        
        data = np.zeros((15), dtype=np.float32)
        pose = self._slam.estimate_ego_pose()
        velocity = self._ego.get_odometer().read()
        #gps = self._ego.get_gps().read()
        imu = self._ego.get_imu().read()

        data[0] = pose.x
        data[1] = pose.y
        data[2] = pose.heading
        data[3] = velocity
        # data[4] = gps.latitude
        # data[5] = gps.longitude
        # data[6] = gps.altitude
        data[7] = imu.accel_x
        data[8] = imu.accel_y
        data[9] = imu.accel_z
        data[10] = imu.gyro_x
        data[11] = imu.gyro_y
        data[12] = imu.gyro_z
        data[13] = 0
        data[14] = imu.heading

        self._sensor_link.write_float_np(data)

class RemoteControlServer(DiscreteComponent):
    _control_link: PyDatalink
    _ego: EgoCar
    _on_mission_restart: callable
    
    def __init__(self,  period_ms: int, ego: EgoCar, on_mission_restart: callable) -> None:
        super().__init__(period_ms)
        self._control_link = PyDatalink(host=None, port=CAR_CONTROL_PORT, timeout_ms=1000)
        self._ego = ego
        self._on_mission_restart = on_mission_restart
    
    def wait_ready(self) -> None:
        while not self._control_link.is_connected():
            time.sleep(0.05)
    
    def _loop(self, dt: float) -> None:
        if not self._control_link.is_connected():
            return
        
        if not self._control_link.has_data():
            return

        control_data = self._control_link.read_flatten_float_np()

        match int(control_data[0]):
            case 1:
                self.__set_autopilot(float(control_data[1]))
            case 2:
                self.__set_power(float(control_data[1]))
            case 3:
                self.__set_brake(float(control_data[1]))
            case 4:
                self.__set_steering(float(control_data[1]))
            case 5:
                self.__on_restart_mission()
            case 6:
                self.__set_pose(control_data)
    
    def __set_autopilot(self, val: float):
        self._ego.set_autopilot(val > 0)

    def __set_power(self, val: float):
        self._ego.set_autopilot(False)
        self._ego.set_autonomous_mode_flag(False)
        self._ego.set_power(val)

    def __set_brake(self, val: float):
        self._ego.set_autopilot(False)
        self._ego.set_autonomous_mode_flag(False)
        self._ego.set_brake(val)

    def __set_steering(self, val: float):
        self._ego.set_autopilot(False)
        self._ego.set_autonomous_mode_flag(False)
        self._ego.set_steering(val)
    
    def __on_restart_mission(self):
        self._on_mission_restart()
        
    def __set_pose(self, data: np.ndarray):
        x = float(data[1])
        y = float(data[2])
        z = float(data[3])
        heading = float(data[4])
        self._ego.set_pose (x, y, z, heading) 

class RemoteDebugServer(DiscreteComponent):
    _debug_link: PyDatalink
    _carla_client: CarlaClient
    
    def __init__(self,  period_ms: int, carla_client: CarlaClient) -> None:
        super().__init__(period_ms)
        self._debug_link = PyDatalink(host=None, port=CAR_DEBUG_PORT, timeout_ms=1000)
        self._carla_client = carla_client

    def wait_ready(self) -> None:
        while not self._debug_link.is_connected():
            time.sleep(0.05)
    
    def _loop(self, dt: float) -> None:
        if not self._debug_link.is_connected():
            return
        
        if not self._debug_link.has_data():
            return

        debug_data = self._debug_link.read_flatten_float_np()
        debug_data = debug_data.reshape(math.floor(debug_data.shape[0] / 3), 3)
        
        type = int(debug_data[0, 0])
        count = int(debug_data[0, 1])

        if type == 1:
            self.__show_point(VehiclePose(
                x=float(debug_data[1, 0]),
                y=float(debug_data[1, 1]),
                heading_angle=debug_data[1, 2],
                desired_speed=0.0
            ))
            return
        
        path: List[VehiclePose] = []
        for i in range (1, count+1):
            path.append(VehiclePose(
                x=float(debug_data[i, 0]),
                y=float(debug_data[i, 1]),
                heading_angle=debug_data[i, 2],
                desired_speed=0.0
            ))

        if type == 2:
            self.__show_path(path)
        elif type == 3:
            self.__show_global_path(path)
        

    def __show_point(self, p: VehiclePose):
        world = self._carla_client.get_world()
        world.debug.draw_string(carla.Location(p.x, p.y, 2), 'X', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=1200.0,
                                persistent_lines=True)

    def __show_path(self, path: List[VehiclePose]):
        world = self._carla_client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=1200.0,
                                    persistent_lines=True)
            
    def __show_global_path(self, path: List[VehiclePose]):
        world = self._carla_client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'X', draw_shadow=False,
                                    color=carla.Color(r=0, g=0, b=255), life_time=1200.0,
                                    persistent_lines=True)
class EgoCarServer:
    _remote_planning_data_sender: PrePlanningDataSender
    _remote_sensors: RemoteSensorsServer
    _remote_control: RemoteControlServer
    _remote_debug: RemoteDebugServer

    def __init__(self, ego: EgoCar, client: CarlaClient, slam: SLAM, on_mission_restart: callable) -> None:
        self._ego = ego
        self._remote_sensors = RemoteSensorsServer(1, ego, slam)
        self._remote_planning_data_sender = PrePlanningDataSender(20, ego, True)
        self._remote_control = RemoteControlServer(0.1, ego, on_mission_restart)
        self._remote_debug = RemoteDebugServer(100, client)

    def start(self) -> None:
        self._remote_sensors.start()
        self._remote_planning_data_sender.start()
        self._remote_control.start()
        self._remote_debug.start()

        self._remote_sensors.wait_ready()
        self._remote_planning_data_sender.wait_ready()
        self._remote_control.wait_ready()
        self._remote_debug.wait_ready()

    def destroy(self) -> None:
        self._remote_sensors.destroy()
        self._remote_planning_data_sender.destroy()
        self._remote_control.destroy()
        self._remote_debug.destroy()
     