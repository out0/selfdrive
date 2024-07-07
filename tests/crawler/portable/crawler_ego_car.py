import sys
import ctypes
sys.path.append("../")
from model.sensors import Gps, Odometer, IMU
from model.camera import Camera
from model.ego_car import EgoCar

LIBNAME = "/usr/local/lib/libcrawler.so"
lib = ctypes.CDLL(LIBNAME)

lib.initialize.restype = ctypes.c_void_p
lib.initialize.argtypes = [ctypes.c_char_p]

lib.resetActuators.restype = ctypes.c_bool
lib.resetActuators.argtypes = [ctypes.c_void_p]

lib.stop.restype = ctypes.c_bool
lib.stop.argtypes = [ctypes.c_void_p]

lib.setPower.restype = ctypes.c_bool
lib.setPower.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.setSteeringAngle.restype = ctypes.c_bool
lib.setSteeringAngle.argtypes = [ctypes.c_void_p, ctypes.c_float]

class CrawlerEgoCar(EgoCar):
    _ego_controller: ctypes.c_void_p
    
    def __init__(self, device: str) -> None:
        super().__init__()
        self._ego_controller = lib.initialize(ctypes.c_char_p(
                device.encode('utf-8')))

    def get_odometer(self) -> Odometer:
        pass

    def get_gps(self) -> Gps:
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
    
    def set_power(self, power: float) -> bool:
        return lib.setPower(self._ego_controller, power)
        pass
    
    def set_brake(self, brake: float) -> bool:
        return lib.stop(self._ego_controller)
        pass
    
    def set_steering(self, angle: int) -> bool:
        return lib.setSteeringAngle(self._ego_controller, angle)
        pass
    