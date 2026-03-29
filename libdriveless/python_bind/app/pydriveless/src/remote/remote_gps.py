from pydatalink import Datalink
import numpy as np
from .. sensors.gps import GPS, GpsData
import time
import threading

GPS_SENSOR_DATA_DEFAULT_PORT = 7704
GPS_SENSOR_DATA_SIZE = 3

class RemoteGPSServer:
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _gps: GPS
    _gps_period_s: float

    def __init__(self, gps: GPS, gps_period_ms: int = 250, port: int = GPS_SENSOR_DATA_DEFAULT_PORT):
        self._running = True
        self._gps = gps
        self._gps_period_s = gps_period_ms / 1000.0
        self._data_link = Datalink(port=port, timeout=1000)
        self._data_thread = threading.Thread(target=self._send_sensor_data)
        self._data_thread.start()
        
    def __del__(self):
        self.terminate()
        
    def terminate(self) -> None:
        self._running = False
        self._data_thread.join()
        if self._data_link is not None:
            del self._data_link
            self._data_link = None          
    
    def _send_sensor_data(self) -> None:
        conn_data = np.zeros(shape=(GPS_SENSOR_DATA_SIZE,), dtype=np.float32)
        while self._running:
            if not self._data_link.is_ready():
                time.sleep(0.01)
                continue

            gps_data = self._gps.read()
            if gps_data is None:
                time.sleep(self._gps_period_s)
                continue

            if gps_data.valid and self._data_link.is_ready():
                conn_data[0] = gps_data.lat
                conn_data[1] = gps_data.lon
                conn_data[2] = gps_data.alt
                self._data_link.write(conn_data)
            time.sleep(self._gps_period_s)
    

class RemoteGPSClient(GPS):
    _data_link: Datalink
    _data_thread: threading.Thread
    _running: bool
    _gps: GPS

    def __init__(self, host: str = "127.0.0.1", port: int = GPS_SENSOR_DATA_DEFAULT_PORT):
        self._running = True
        self._last_gps_data = None
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

            data, size, timestamp = self._data_link.read_np(shape=(GPS_SENSOR_DATA_SIZE,), dtype=np.float32)
            if size == 0:
                print ("zero size GPS")
                continue

            self._last_gps_data = GpsData(
                lat=data[0],
                lon=data[1],
                alt=data[2],
                timestamp=timestamp,
                valid=True)
            
            self._data_link.write_keep_alive()
    
    def read(self) -> GpsData:
        return self._last_gps_data
