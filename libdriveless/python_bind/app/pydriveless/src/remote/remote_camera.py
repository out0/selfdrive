from pydatalink import Datalink
import numpy as np
from ..sensors.camera import Camera
import time
import threading

CAMERA_SENSOR_DATA_DEFAULT_PORT = 22100
CAMERA_SENSOR_INFO_DEFAULT_PORT = 22110


class RemoteCameraServer:
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _camera: Camera
    _camera_period_s: float
    _camera_info_mode: bool
    _img_shape: tuple[int, ...]
    _img_dtype: any

    def __init__(self, camera: Camera, period_ms: int = 10, port: int = CAMERA_SENSOR_DATA_DEFAULT_PORT):
        self._running = True
        self._camera = camera
        self._camera_period_s = period_ms / 1000.0
        self._data_link = Datalink(port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._send_sensor_data)
        self._data_thread.start()
        self._camera_info_mode = False
        self._img_shape = None
        self._img_dtype = None

    def __del__(self):
        self._running = False
        self._data_thread.join()
        del self._data_link


    def encode_dtype(self, dtype: np.dtype) -> int:
        dtype = np.dtype(dtype)

        if dtype == np.bool_:
            return 0

        # Unsigned integers
        elif dtype == np.uint8:
            return 1
        elif dtype == np.uint16:
            return 2
        elif dtype == np.uint32:
            return 3
        elif dtype == np.uint64:
            return 4

        # Signed integers
        elif dtype == np.int8:
            return 5
        elif dtype == np.int16:
            return 6
        elif dtype == np.int32:
            return 7
        elif dtype == np.int64:
            return 8

        # Floats
        elif dtype == np.float16:
            return 9
        elif dtype == np.float32:
            return 10
        elif dtype == np.float64:
            return 11

        else:
            return -1

    def _send_info_data(self) -> None:
        if self._img_shape is None:
            img, _ = self._camera.read()
            if img is not None:
                self._img_shape = img.shape
                self._img_dtype = img.dtype
            return
        
        
        info = np.array([self._img_shape[0],
                        self._img_shape[1],
                        self._img_shape[2],
                        self._camera.fov(),
                        self._camera.fps(),
                        self.encode_dtype(self._img_dtype)], dtype=np.int32)
        self._data_link.write(info, timestamp=1)
    
    def _check_link_mode(self) -> None:
        if not self._data_link.has_data():
            return

        print("received link mode command?")

        mode_data, size, _ = self._data_link.read_np(shape=(1,), dtype=np.int32)
        if size == 0:
            return None
        
        print(f"received link mode command: {mode_data}")

        mode = mode_data[0]
        if mode == 1:
            self._camera_info_mode = True
        else:
            self._camera_info_mode = False
    
    def _send_sensor_data(self) -> None:
        while self._running:
            img, _ = self._camera.read()
            if img is None:
                continue

            if not self._data_link.is_ready():
                time.sleep(0.01)
                continue

            self._check_link_mode()

            if self._camera_info_mode:
                print ("Sending camera info data")
                self._send_info_data()
            else:
                print ("Sending IMAGE data")
                self._data_link.write(img)
            
            time.sleep(self._camera_period_s)

   

class RemoteCameraClient(Camera):
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _width: int
    _height: int
    _fov: int
    _fps: int
    _dtype: np.dtype
    _last_camera_data: np.ndarray
    _last_camera_data_timestamp: float

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = CAMERA_SENSOR_DATA_DEFAULT_PORT):
        self._host = host
        self._running = True
        self._last_camera_data = None
        self._last_camera_data_timestamp = 0.0
        self._shape = None
        self._dtype = None
        self._width = 0
        self._height = 0
        self._fov = 0
        self._fps = 0
        self._data_link = Datalink(host=host, port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._read_sensor_data)
        self._data_thread.start()

    def __del__(self):
        self._running = False
        self._data_thread.join()
        del self._data_link

    def decode_dtype(self, code: int) -> np.dtype:
        if code == 0:
            return np.bool_
        elif code == 1:
            return np.uint8
        elif code == 2:
            return np.uint16
        elif code == 3:
            return np.uint32
        elif code == 4:
            return np.uint64
        elif code == 5:
            return np.int8
        elif code == 6:
            return np.int16
        elif code == 7:
            return np.int32
        elif code == 8:
            return np.int64
        elif code == 9:
            return np.float16
        elif code == 10:
            return np.float32
        elif code == 11:
            return np.float64
        else:
            return None  # or raise ValueError("Unknown dtype code")

    def _set_camera_to_info_mode(self, info_mode: bool) -> None:
        mode = np.array([1], dtype=np.int32)
        mode[0] = 1 if info_mode else 0  # Info mode
        self._data_link.write(mode)

    def _check_camera_is_in_info_mode(self) -> bool:
        next_message_size = self._data_link.next_message_size()
        return next_message_size == np.dtype(np.int32).itemsize * 6

    def _receive_camera_info(self) -> None:
        if not self._check_camera_is_in_info_mode():
            self._set_camera_to_info_mode(True)
            time.sleep(0.10)
            self._data_link.clear_buffer()            
            return
        
        print (f"reading camera info")
        info, size, timestamp = self._data_link.read_np(shape=(6,), dtype=np.int32)
        if size == 0:
            print (f"reading camera info: empty")
            return
        
        print (f"reading camera info: {info}, size={size}, timestamp={timestamp}")

        self._height = info[0]
        self._width = info[1]
        self._shape = (info[0], info[1], info[2])
        self._fov = info[3]
        self._fps = info[4]
        self._dtype = self.decode_dtype(info[5])

        print (f"setting info mode to false")
        self._set_camera_to_info_mode(False)       
        time.sleep(0.10)
        self._data_link.clear_buffer()


    def _read_sensor_data(self) -> None:
        while self._running:
            if not self._data_link.is_ready():
                print ("data link not ready")
                while not self._data_link.is_ready() and self._running:
                    time.sleep(0.01)
                print ("data link connected")

            if not self._data_link.has_data():
                time.sleep(0.01)
                continue

            self._data_link.write_keep_alive()

            if self._shape is None or self._dtype is None:
                self._receive_camera_info()
                continue
            
            if self._check_camera_is_in_info_mode():
                print (f"camera is still in info mode, requesting data mode")
                self._set_camera_to_info_mode(False)
                self._data_link.clear_buffer()
                time.sleep(0.10)
                continue

            self._last_camera_data, _, self._last_camera_data_timestamp = self._data_link.read_np(
                shape=self._shape, dtype=self._dtype)

    def read(self) -> tuple[np.ndarray, float]:
        return self._last_camera_data, self._last_camera_data_timestamp

    def fov(self) -> int:
        return self._fov

    def fps(self) -> int:
        return self._fps

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height
