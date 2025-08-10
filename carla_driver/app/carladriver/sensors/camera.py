import carla
import math, random
from threading import Thread, Lock
from pydriveless import Camera
import time
from .. control.session import CarlaSession
from .. control.ego import EgoVehicle
import numpy as np
import weakref
from pydriveless import Telemetry

TELEMETRY = True

class CarlaCamera(Camera):
    __camera_obj: any
    __width: int
    __height: int
    __fov: float
    __fps: int
    __pos: tuple[float, float, float]
    __rotation: tuple[float, float, float]
    __last_frame: np.ndarray
    __mtx: Lock
    __timestamp: float
    
    def __init__(self, 
                 session: CarlaSession,
                 vehicle_obj: any,
                 width: int,
                 height: int,
                 fov: float = 120.0,
                 fps: int = 30,
                 pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 camera_type: str = "sensor.camera.rgb"                    
                 ):
            self.__width = width
            self.__height = height
            self.__fov = fov
            self.__fps = fps
            self.__pos = pos
            self.__rotation = rotation
            self.__last_frame = None
            self.__mtx = Lock()
            self.__timestamp = 0.0
            camera_bp = session.world.get_blueprint_library().find(camera_type)
            camera_bp.set_attribute('image_size_x', str(self.__width))
            camera_bp.set_attribute('image_size_y', str(self.__height))
            camera_bp.set_attribute('fov', str(self.__fov))
            camera_bp.set_attribute('sensor_tick', str(1/self.__fps))
            location = carla.Location(x=self.__pos[0], y=self.__pos[1], z=self.__pos[2])
            rotation = carla.Rotation(pitch=self.__rotation[0], yaw=self.__rotation[1], roll=self.__rotation[2])
            camera_transform = carla.Transform(location, rotation)
            self.__camera_obj = session.client.get_world().spawn_actor(camera_bp, camera_transform, attach_to=vehicle_obj)
            #self.__camera_obj.listen(self.test)
            weak_self = weakref.ref(self)
            self.__camera_obj.listen(lambda p: CarlaCamera.__new_data(weak_self, p))

    # def test(self, raw) -> None:
    #     if raw is None:
    #         return
    #     print ("got camera data")
    #     array = np.frombuffer(raw.raw_data, dtype=np.uint8)
    #     array = np.reshape(array, (raw.height, raw.width, 4))
    #     array = array[:, :, :3]
    #     array = array[:, :, ::-1]
    #     Telemetry.log_if(TELEMETRY, f"log/raw_camera.png", array, append=False)

    @staticmethod
    def __to_bgra_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    @staticmethod
    def __to_rgb_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a RGB numpy array."""
        array = CarlaCamera.__to_bgra_array(image)
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    @staticmethod
    def __new_data(weak_self, raw_data: any) -> None:
        if raw_data is None:
            return
        self = weak_self()
        if not self.__mtx.acquire(blocking=False): return
        self.__last_frame = CarlaCamera.__to_rgb_array(raw_data)
        self.__timestamp = time.time()
        self.__mtx.release()
       
        
    
    def read(self) -> tuple[np.ndarray, float]:
        if not self.__mtx.acquire(blocking=True): 
            return None, 0.0
        frame = self.__last_frame
        timestamp = self.__timestamp
        self.__mtx.release()
        return frame, timestamp
    
    def destroy(self) -> None:
        if self.__camera_obj is None:
            return
        
        self.__camera_obj.destroy()
        self.__camera_obj = None
        
    def width(self) -> int:
        return self.__width

    def height(self) -> int:
        return self.__height


class BevCameraBase(CarlaCamera):
    Z_DIST = 10.0
    FOV = 120.0
    """
    A class representing a Bird's Eye View (BEV) camera in CARLA.
    It inherits from CarlaCamera and is used to capture images from a top-down perspective.
    """
    def __init__(self, session: CarlaSession, vehicle: any, width: int, height: int, fps: int = 30, camera_type="sensor.camera.rgb"):
        super().__init__(session, vehicle, width, height, BevCameraBase.FOV, fps, pos=(0.0, 0.0, BevCameraBase.Z_DIST), rotation=(-90.0, 0.0, 0.0), camera_type=camera_type)
    
    def __real_size() -> float:
        return 2* BevCameraBase.Z_DIST*math.tan(math.radians(BevCameraBase.FOV/2))
    
    def real_width(self) -> float:
        return BevCameraBase.__real_size()
    
    def real_height(self) -> float:
        return BevCameraBase.__real_size()
    
class BevCamera(BevCameraBase):
    """
    A class representing a Bird's Eye View (BEV) camera in CARLA.
    It inherits from BevCameraBase and is used to capture images from a top-down perspective.
    """
    def __init__(self, session: CarlaSession, vehicle_obj: any, width: int, height: int, fps: int = 30):
        super().__init__(session, vehicle_obj, width, height, fps, camera_type="sensor.camera.rgb")
        
class BevCameraSemantic(BevCameraBase):
    """
    A class representing a Bird's Eye View (BEV) camera in CARLA.
    It inherits from BevCameraBase and is used to capture segmented ground-truth images from a top-down perspective.
    """
    def __init__(self, session: CarlaSession, vehicle_obj: any, width: int, height: int, fps: int = 30):
        super().__init__(session, vehicle_obj, width, height, fps, camera_type="sensor.camera.semantic_segmentation")

