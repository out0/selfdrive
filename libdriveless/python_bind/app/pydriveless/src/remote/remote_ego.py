from pydatalink import Datalink
import numpy as np
from .. ego_vehicle import EgoVehicle
import time
import threading

CONTROL_DATA_DEFAULT_PORT = 22001
CONTROL_DATA_SIZE = 2

CMD_SET_THROTTLE = 1.0
CMD_SET_STEERING = 2.0
CMD_SET_BRAKE = 3.0

class RemoteEgoServer:
    _control_link: Datalink
    _control_thread: threading.Thread
    _running: bool
    _ego: EgoVehicle

    def __init__(self, ego: EgoVehicle, port: int = CONTROL_DATA_DEFAULT_PORT):
        self._running = True
        self._ego = ego
        self._control_link = Datalink(port=port, timeout=1000)
        self._control_thread = threading.Thread(target=self._read_control_data)
        self._control_thread.start()
    
    def __del__(self):
        self._running = False
        self._control_thread.join()
        del self._control_link
    
    def _read_control_data(self) -> None:
        while self._running:
            if not self._control_link.has_data():
                time.sleep(0.01)
                continue
            
            data, size = self._control_link.read_np(shape=(CONTROL_DATA_SIZE,), dtype=np.float32)
            if size == 0:
                continue

            cmd_type = data[0]
            cmd_val = data[1]

            if cmd_type == CMD_SET_THROTTLE:  # control command
                self._ego.set_power(cmd_val)
            elif cmd_type == CMD_SET_STEERING:
                self._ego.set_steering(cmd_val)
            elif cmd_type == CMD_SET_BRAKE:
                self._ego.set_brake(cmd_val)
            
            #self._control_link.write(np.array([cmd_type], dtype=np.float32))  # ack


class RemoteEgoClient(EgoVehicle):
    _control_link: Datalink
    _running: bool

    def __init__(self, host: str = "127.0.0.1", port: int = CONTROL_DATA_DEFAULT_PORT):
        self._running = True
        self._control_link = Datalink(host=host, port=port, timeout=1000)
    
    def __del__(self):
        self._running = False
        del self._control_link
    
    def set_power(self, power_level: float) -> None:
        cmd = np.array([CMD_SET_THROTTLE, power_level], dtype=np.float32)
        self._control_link.write(cmd)
    
    def set_brake(self, brake_level: float) -> None:
        cmd = np.array([CMD_SET_BRAKE, brake_level], dtype=np.float32)
        self._control_link.write(cmd)
    
    def set_steering(self, angle: float) -> None:
        cmd = np.array([CMD_SET_STEERING, angle], dtype=np.float32)
        self._control_link.write(cmd)

