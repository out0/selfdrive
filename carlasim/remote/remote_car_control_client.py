from typing import List

import numpy as np
from model.map_pose import VehiclePose
from model.discrete_component import DiscreteComponent
from model.planning_data import PlanningData, PrePlanningData
from slam.slam import SLAM
from utils.datalink.pybind.pydatalink import PyDatalink
import time, math
from utils.float_encoder import FloatEncoder
from threading import Lock



PRE_PLANNING_DATA_PORT = 19990
SENSOR_PORT = 19991
CAR_CONTROL_PORT = 19992
CAR_DEBUG_PORT = 19993

RECV_FRAME_W = 256
RECV_FRAME_H = 256
RECV_BEV_W = 256
RECV_BEV_H = 256

class PrePlanningDataReceiver(DiscreteComponent):
    _datalink: PyDatalink
    _pose: VehiclePose
    _frame_front: np.ndarray
    _frame_left: np.ndarray
    _frame_right: np.ndarray
    _frame_back: np.ndarray
    _frame_bev: np.ndarray
    _velocity: float
    _lock: Lock
    _frames_size: int
    _sensor_data_size: int
    _recv_as_bev: bool

    def __init__(self, period_ms: int, host: str, recv_as_bev: bool) -> None:
        super().__init__(period_ms)
        self._datalink = PyDatalink(host=host, port=PRE_PLANNING_DATA_PORT, timeout_ms=1000)
        self._pose = None
        self._velocity = 0
        self._frame = None
        self._lock = Lock()

        if recv_as_bev:
            self._frames_size = 3 * RECV_BEV_W * RECV_BEV_H
        else:
            self._frames_size = 4 * 3 * RECV_FRAME_W * RECV_FRAME_H

        self._sensor_data_size = 80
        self._recv_as_bev = recv_as_bev

        self._frame_front = None
        self._frame_left = None
        self._frame_right = None
        self._frame_back = None
        self._frame_bev = None
    
    def wait_ready(self) -> None:
        while not self._datalink.is_connected():
            time.sleep(0.05)

    def _loop(self, dt: float) -> None:
        if not self._datalink.is_connected():
            return

        self._lock.acquire(True)

        if not self._datalink.has_data():
            self._lock.release()
            return

        data = self._datalink.read_uint8_np((self._sensor_data_size + self._frames_size))

        if data is None:
            return

        sensor_data = data[0: self._sensor_data_size].reshape((10, 8))


        if self._recv_as_bev:
            self._frame_bev = data[self._sensor_data_size : ].reshape((RECV_BEV_W, RECV_BEV_H, 3))
            
        else:
            frames_data = data[self._sensor_data_size : ].reshape((4, RECV_FRAME_W, RECV_FRAME_H, 3))
            (self._frame_front, 
            self._frame_left, 
            self._frame_right, 
            self._frame_back) = np.split(frames_data, 4, axis=0)

        poseX = FloatEncoder.decode(bytes(sensor_data[0, 0:4]))
        poseY = FloatEncoder.decode(bytes(sensor_data[1, 0:4]))
        heading = FloatEncoder.decode(bytes(sensor_data[2, 0:4]))
        velocity = FloatEncoder.decode(bytes(sensor_data[3, 0:4]))
        # gpsLat = FloatEncoder.decode(bytes(sensor_data[4, :]))
        # gpsLon = FloatEncoder.decode(bytes(sensor_data[5, :]))
        # gpsAlt = FloatEncoder.decode(bytes(sensor_data[6, :]))

        self._pose = VehiclePose(
            x=poseX,
            y=poseY,
            heading_angle=heading,
            desired_speed=0
        )

        self._velocity = velocity
        self._lock.release()
    
    def get_planning_data(self) -> PlanningData:
        if self._recv_as_bev:
            return PlanningData(self._frame_bev, self._pose, self._velocity)
        else:
            raise Exception("should call get_pre_planning_data() since you've opted to receive frames as BEV")

    def get_pre_planning_data(self) -> PrePlanningData:
        if self._recv_as_bev:
            return PrePlanningData(self._pose, self._velocity, self._frame_front, self._frame_back, self._frame_left, self._frame_right)
        else:
            raise Exception("should call get_pre_planning_data() since you've opted to receive frames as BEV")
        
class RemoteSensorsClient(DiscreteComponent):
    _datalink: PyDatalink
    _pose: VehiclePose
    _velocity: float
    _lock: Lock

    def __init__(self, period_ms: int, host: str) -> None:
        super().__init__(period_ms)
        self._datalink = PyDatalink(host=host, port=SENSOR_PORT, timeout_ms=1000)
        self._pose = None
        self._velocity = 0
        self._lock = Lock()
    
    def wait_ready(self) -> None:
        while not self._datalink.is_connected() or self.get_pose() is None:
            time.sleep(0.05)

    def wait_start(self) -> None:
        print ("[Ego vision] waiting for server to come up")
        while (not self._datalink.is_connected()):
            time.sleep(0.5)


    def _loop(self, dt: float) -> None:
        if not self._datalink.has_data():
            return
        
        self._lock.acquire(True)
        data = self._datalink.read_float_np((15))
 
        self._pose = VehiclePose(
            x=data[0],
            y=data[1],
            heading_angle=data[2],
            desired_speed=0
        )

        self._velocity = data[3]
        self._lock.release()

    def is_connected(self) -> bool:
        return self._datalink.is_connected()
    
    def get_pose(self) -> VehiclePose:
        self._lock.acquire(True)
        pose = self._pose
        self._lock.release()
        return pose

    def get_velocity(self) -> float:
        self._lock.acquire(True)
        vel = self._velocity
        self._lock.release()
        return vel
    
class RemoteControlClient:
    _control_link: PyDatalink
    _debug_link: PyDatalink
    _command_arr: np.ndarray
    _debug_arr: np.ndarray
    
    def __init__(self, host: str) -> None:
        self._control_link = PyDatalink(host=host, port=CAR_CONTROL_PORT, timeout_ms=1000)
        self._debug_link = PyDatalink(host=host, port=CAR_DEBUG_PORT, timeout_ms=1000)
        self._command_arr = np.zeros((5), dtype=np.float32)
        self._debug_arr = np.zeros((30, 3), dtype=np.float32)
    
    def wait_ready(self) -> None:
        while not self._debug_link.is_connected() or not self._control_link.is_connected():
            time.sleep(0.05)

    def set_autopilot(self, value: bool) -> None:
        i = 0
        if value:
            i = 1
        self._command_arr[0] = 1
        self._command_arr[1] = i
        self._control_link.write_float_np(self._command_arr)
        
    def set_power(self, power: float) -> None:
        self._command_arr[0] = 2
        self._command_arr[1] = power
        self._control_link.write_float_np(self._command_arr)

    def set_brake(self, brake_val: float) -> None:
        self._command_arr[0] = 3
        self._command_arr[1] = brake_val
        self._control_link.write_float_np(self._command_arr)

    def set_steering(self, angle: int) -> None:
        self._command_arr[0] = 4
        self._command_arr[1] = angle
        self._control_link.write_float_np(self._command_arr)

        
    def restart_mission(self) -> None:
        self._command_arr[0] = 5
        self._control_link.write_float_np(self._command_arr)
    
    def set_pose(self, x: int, y: int, z: int, heading: int) -> None:
        self._command_arr[0] = 6
        self._command_arr[1] = x
        self._command_arr[2] = y
        self._command_arr[3] = z
        self._command_arr[4] = heading
        self._control_link.write_float_np(self._command_arr)
        
    def __increase_debug_buffer(self) -> None:
        self._debug_arr = np.zeros((2*self._debug_arr.shape[0], 3), dtype=np.float32)

    def show_point(self, point: VehiclePose):
        self._debug_arr[0, 0] = 1
        self._debug_arr[0, 1] = 1
        self._debug_arr[1, 0] = point.x
        self._debug_arr[1, 1] = point.y
        self._debug_arr[1, 2] = point.heading
        self._debug_link.write_float_np(self._debug_arr)

    def show_path(self, path: List[VehiclePose]):
        while len(path) > self._debug_arr.shape[0]:
            self.__increase_debug_buffer()
        
        self._debug_arr[0, 0] = 2
        self._debug_arr[0, 1] = len(path)
        pos: int = 1
        for point in path:
            self._debug_arr[pos, 0] = point.x
            self._debug_arr[pos, 1] = point.y
            self._debug_arr[pos, 2] = point.heading
            pos += 1

        self._debug_link.write_float_np(self._debug_arr)

    def show_global_path(self, path: List[VehiclePose]):
        while len(path) > self._debug_arr.shape[0]:
            self.__increase_debug_buffer()
        
        self._debug_arr[0, 0] = 3
        self._debug_arr[0, 1] = len(path)
        pos: int = 1
        for point in path:
            self._debug_arr[pos, 0] = point.x
            self._debug_arr[pos, 1] = point.y
            self._debug_arr[pos, 2] = point.heading
            pos += 1

        self._debug_link.write_float_np(self._debug_arr)


class RemoteOdometer:
    _sensors_client: RemoteSensorsClient

    def __init__(self, sensors_client: RemoteSensorsClient) -> None:
        super().__init__()
        self._sensors_client = sensors_client

    def read(self) -> float:
        return self._sensors_client.get_velocity()

class EgoCarClient:
    _remote_planning_data_receiver: PrePlanningDataReceiver
    _remote_sensors: RemoteSensorsClient
    _remote_control_and_debug: RemoteControlClient

    def __init__(self, host: str, recv_as_bev: bool) -> None:
        self._remote_planning_data_receiver = PrePlanningDataReceiver(20, host, recv_as_bev)
        self._remote_sensors = RemoteSensorsClient(1, host)
        self._remote_control_and_debug = RemoteControlClient(host)
        
    def start(self) -> None:
        self._remote_planning_data_receiver.start()
        self._remote_sensors.start()
        
        self._remote_planning_data_receiver.wait_ready()
        self._remote_sensors.wait_ready()
        self._remote_control_and_debug.wait_ready()



    def destroy(self) -> None:
        self._remote_planning_data_receiver.destroy()
        self._remote_sensors.destroy()

    def set_autopilot(self, value: bool) -> None:
        self._remote_control_and_debug.set_autopilot(value)
        
    def set_power(self, power: float) -> None:
        self._remote_control_and_debug.set_power(power)

    def set_brake(self, brake_val: float) -> None:
        self._remote_control_and_debug.set_brake(brake_val)

    def set_steering(self, angle: int) -> None:
        self._remote_control_and_debug.set_steering(angle)

    def show_global_path(self, path: List[VehiclePose]):
        self._remote_control_and_debug.show_global_path(path)

    def show_path(self, path: List[VehiclePose]):
        self._remote_control_and_debug.show_path(path)

    def get_odometer(self) -> RemoteOdometer:
        return RemoteOdometer(self._remote_sensors)
    
    def get_planning_data(self) -> PlanningData:
        return self._remote_planning_data_receiver.get_planning_data()
    
    def get_pre_planning_data(self) -> PlanningData:
        return self._remote_planning_data_receiver.get_pre_planning_data()
    
    def restart_mission(self) -> None:
        self._remote_control_and_debug.restart_mission()