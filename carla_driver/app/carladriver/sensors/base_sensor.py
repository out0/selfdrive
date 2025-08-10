import carla
import threading
import time
from .. control.session import CarlaSession
import weakref

class CarlaSensor:
    _sensor: any
    _raw_data: any
    _lock: threading.Lock
    _lock_max_wait: float
    _last_read_timestamp: float

    def __init__(self, 
                 type: str, 
                 session: CarlaSession, 
                 vehicle: any, 
                 period_ms: int,
                 pos: tuple[float, float, float],
                 rotation: tuple[float, float, float],
                 custom_attributes: dict = None):
        #sensor_bp = session.blueprint_lib.find(type)
        sensor_bp = session.blueprint_lib.filter(type)[0]
        period_s = period_ms / 1000
        self._lock_max_wait = 4 * period_s
        sensor_bp.set_attribute('sensor_tick', str(period_s))
        for key, value in (custom_attributes or {}).items():
            sensor_bp.set_attribute(key, str(value))
        if pos is None and rotation is None:
            transform = carla.Transform()
        else:            
            transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=pos[2]), carla.libcarla.Rotation(rotation[0], rotation[1], rotation[2]))
            
        self._sensor = session.world.spawn_actor(sensor_bp, transform, attach_to=vehicle)
        self._lock = threading.Lock()
        self._last_read_timestamp = -1
        self._raw_data = None
        
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda p: CarlaSensor.__new_data(weak_self, p))

    @staticmethod
    def __new_data(weak_self, raw_data: any) -> None:
        self = weak_self()
        if not self._lock.acquire(blocking=True, timeout=self._lock_max_wait): return
        self._raw_data = raw_data
        self._last_read_timestamp = time.time()
        self._lock.release()
        
    def read(self) -> tuple[any, float]:
        if not self._lock.acquire(blocking=True, timeout=self._lock_max_wait): 
           return None, 0.0
        data = self._raw_data
        self._lock.release()
        return data, self._last_read_timestamp