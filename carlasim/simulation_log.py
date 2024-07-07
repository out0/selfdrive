import threading, time, numpy as np, math
from carlasim.carla_ego_car import EgoCar
from carlasim.sensors.carla_camera import *
    
class AutoTrigger:
    _ego_car: EgoCar
    _trigger_dist: float
    _trigger_time: int
    _trigger_by_time_thr : threading.Thread
    _trigger_by_distance_thr : threading.Thread
    _running: bool
    _last_pose: np.array
    _method: callable
    
    def __init__(self, ego_car: EgoCar, method: callable, trigger_dist: float = 0, trigger_time_ms: int = 0) -> None:
        self._ego_car = ego_car
        self._method = method
        self._trigger_dist = trigger_dist
        self._trigger_time = trigger_time_ms
        self._log_by_time_thr = None
        self._log_by_distance_thr = None
        self._running = False
        
        if trigger_time_ms > 0:
            self._log_by_time_thr = threading.Thread(target=self.__time_trigger_handler)
            self._log_by_time_thr.start()

        if trigger_dist > 0:
            self._log_by_distance_thr = threading.Thread(target=self.__distance_trigger_handler)
            self._log_by_distance_thr.start()

    def __time_trigger_handler(self) -> None:
        self._running = True

        while (self._running):
            time.sleep(self._trigger_time / 1000)
            self._method()
    
    def __compute_distance(self, l: np.array, l2: np.array):
        dx = l2[0] - l[0]
        dy = l2[1] - l[1]
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def __distance_trigger_handler(self) -> None:
        self._running = True
        self._method()
        self._last_pose = self.__get_location()

        while (self._running):
            l = self.__get_location()
            if self.__compute_distance(self._last_pose, l) >= self._trigger_dist:
                self._method()
                self._last_pose = self.__get_location()
                time.sleep(0.001)
    
    def __get_location(self) -> np.array:
        l = np.zeros([3])
        location = self._ego_car.get_carla_ego_car_obj().get_location()
        l[0] = location.x
        l[1] = location.y
        l[2] = location.z
        return l

    def destroy(self) -> None:
        self._running = False

class SimulationPositionLogger:
    _out_file: str
    _ego_car: EgoCar
    _auto_trigger: AutoTrigger

    def __init__(self, ego_car: EgoCar, out_file: str, trigger_time_ms: int = None, trigger_distance_m: int = None) -> None:
        self._ego_car = ego_car
        self._out_file = out_file
        self._auto_trigger = AutoTrigger(ego_car=ego_car,
                                         method=self.log_pose,
                                         trigger_time_ms=trigger_time_ms,
                                         trigger_dist=trigger_distance_m)

    def log_pose(self) -> None:
        f = open(self._out_file, "a")
        o = self._ego_car.get_carla_ego_car_obj()
        location = o.get_location()
        yaw = o.get_transform().rotation.yaw
        f.write(f"{location.x};{location.y};{location.z};{yaw}\n")
        f.close()
