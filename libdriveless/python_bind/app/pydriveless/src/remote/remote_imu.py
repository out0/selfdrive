from pydatalink import Datalink
import numpy as np
from .. sensors.imu import IMU, IMUData
import time
import threading

IMU_SENSOR_DATA_DEFAULT_PORT = 22003
IMU_SENSOR_DATA_SIZE = 7

CMD_SET_THROTTLE = 1.0
CMD_SET_STEERING = 2.0
CMD_SET_BRAKE = 3.0


class RemoteIMUServer:
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _imu: IMU
    _imu_period_s: float

    def __init__(self, imu: IMU, imu_period_ms: int = 10, port: int = IMU_SENSOR_DATA_DEFAULT_PORT):
        self._running = True
        self._imu = imu
        self._imu_period_s = imu_period_ms / 1000.0
        self._data_link = Datalink(port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._send_sensor_data)
        self._data_thread.start()

    def __del__(self):
        self._running = False
        self._data_thread.join()
        del self._data_link

    def _send_sensor_data(self) -> None:
        while self._running:
            if not self._data_link.is_ready():
                time.sleep(0.01)
                continue

            conn_data = np.zeros(IMU_SENSOR_DATA_SIZE, dtype=np.float32)
            while self._running:
                imu_data = self._imu.read()
                if imu_data.valid and self._data_link.is_ready():
                    conn_data[0] = imu_data.accel_x
                    conn_data[1] = imu_data.accel_y
                    conn_data[2] = imu_data.accel_z
                    conn_data[3] = imu_data.compass
                    conn_data[4] = imu_data.gyro_x
                    conn_data[5] = imu_data.gyro_y
                    conn_data[6] = imu_data.gyro_z
                    self._data_link.write(conn_data)
                time.sleep(self._imu_period_s)


class RemoteIMUClient(IMU):
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _imu: IMU

    def __init__(self, host: str = "127.0.0.1", port: int = IMU_SENSOR_DATA_DEFAULT_PORT):
        self._running = True
        self._last_imu_data = None
        self._data_link = Datalink(host=host, port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._read_sensor_data)
        self._data_thread.start()

    def __del__(self):
        self._running = False
        self._data_thread.join()
        del self._data_link

    def _read_sensor_data(self) -> None:
        while self._running:
            if not self._data_link.is_ready() or not self._data_link.has_data():
                time.sleep(0.01)
                continue

            data, size, timestamp = self._data_link.read_np(
                shape=(IMU_SENSOR_DATA_SIZE,), dtype=np.float32)
            if size == 0:
                continue

            self._last_imu_data = IMUData(
                accel_x=data[0], accel_y=data[1], accel_z=data[2],
                compass=data[3],
                gyro_x=data[4], gyro_y=data[5], gyro_z=data[6],
                valid=True,
                timestamp=timestamp
            )

            self._data_link.write_keep_alive()

    def read(self) -> IMUData:
        return self._last_imu_data
