from pydatalink import Datalink
import numpy as np
from .. sensors.camera import Camera
import time
import threading

CAMERA_SENSOR_DATA_DEFAULT_PORT = 22100
CAMERA_SENSOR_INFO_DEFAULT_PORT = 22110


class RemoteCameraServer:
    _data_link: Datalink
    _info_link: Datalink
    _data_thread: threading.Thread
    _info_thread: threading.Thread
    _running: bool
    _camera: Camera
    _camera_period_s: float

    def __init__(self, camera: Camera, period_ms: int = 10, port: int = CAMERA_SENSOR_DATA_DEFAULT_PORT, info_port: int = CAMERA_SENSOR_INFO_DEFAULT_PORT):
        self._running = True
        self._camera = camera
        self._camera_period_s = period_ms / 1000.0
        self._data_link = Datalink(port=port, timeout=1000)
        self._info_link = Datalink(port=info_port, timeout=1000)
        self._data_thread = threading.Thread(target=self._send_sensor_data)
        self._info_thread = threading.Thread(target=self._send_info_data)
        self._data_thread.start()
        self._info_thread.start()

    def __del__(self):
        self._running = False
        self._data_thread.join()
        self._info_thread.join()
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

    def _send_sensor_data(self) -> None:
        self._camera_info_acked = False

        while self._running:
            if not self._data_link.is_ready():
                self._camera_info_acked = False
                time.sleep(0.01)
                continue

            img, _ = self._camera.read()
            if img is None:
                continue

            self._data_link.write(img)
            time.sleep(self._camera_period_s)

    def _send_info_data(self) -> None:

        shape = None

        while self._running:
            if not self._info_link.is_ready():
                time.sleep(self._camera_period_s)
                continue

            if shape is None:
                img, _ = self._camera.read()
                if img is None:
                    time.sleep(self._camera_period_s)
                    continue
                shape = img.shape

            info = np.array([shape[0],
                             shape[1],
                             shape[2],
                             self._camera.fov(),
                             self._camera.fps(),
                             self.encode_dtype(img.dtype)
                             ], dtype=np.int32)

            self._info_link.write(info)
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


    def _read_sensor_data(self) -> None:
        while self._running:
            if not self._data_link.is_ready() or not self._data_link.has_data():
                time.sleep(0.01)
                continue

            if self._shape is None or self._dtype is None:
                info_link = Datalink(
                    host=self._host, port=CAMERA_SENSOR_INFO_DEFAULT_PORT, timeout=1000)
                while not info_link.has_data():
                    time.sleep(0.01)

                info, size, timestamp = info_link.read_np(shape=(6,), dtype=np.int32)
                if size == 0:
                    continue

                self._height = info[0]
                self._width = info[1]
                self._shape = (info[0], info[1], info[2])
                self._fov = info[3]
                self._fps = info[4]
                self._dtype = self.decode_dtype(info[5])
                info_link.__del__()
                continue

            if self._dtype is None:
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
