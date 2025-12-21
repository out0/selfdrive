
import threading
from collections import deque
import numpy as np
import cv2
from pydriveless import SearchFrame
from pydatalink import Datalink
import time, json

class TelemetryPackage:
    id: str
    data: any
    
    def __init__(self, id: str, data: any):
        self.id = id
        self.data = data

class Telemetry:
    _data_queue: deque
    _write_log_thr: threading.Thread
    _run: bool
    _max_items: int
    _link: Datalink

    def setup(
        port: int,
        max_queued_items: int
    ):
        Telemetry._link = Datalink(port=port, timeout=100, max_incommming_messages_in_queue=1)
        Telemetry._max_items = max_queued_items
        Telemetry._run = True
        Telemetry._data_queue = deque()
        Telemetry._write_log_thr = threading.Thread(target=Telemetry.write_log_handler, daemon=True)
        Telemetry._write_log_thr.start()

    def log_if(condition: bool, id: str, data: any):
        if not condition: return
        Telemetry.log(id, data)

    def log(id: str, data: any):
        if len(Telemetry._data_queue) >= Telemetry._max_items:
            return
        Telemetry._data_queue.append((id, data))
    
    def set_max_items(val: int):
        Telemetry._max_items = val

    def is_empty():
        empty = len(Telemetry._data_queue) == 0
        return empty

    def write_log_handler():
        while Telemetry._run:
            if not Telemetry._link.is_ready():
                #print ("[Telemetry] waiting for the client to connect")
                while not Telemetry._link.is_ready():
                    time.sleep(0.001)
                    continue
                #print ("[Telemetry] client connected")

            c = len(Telemetry._data_queue)
            while c > 0:
                item = Telemetry._data_queue.popleft()
                if item is None: 
                    print ("[Telemetry] null queued item")
                    continue   
                id, data = item

                if isinstance(data, SearchFrame):
                    f = data.get_frame()
                    Telemetry._link.send_object(TelemetryPackage(id, f.astype(np.uint8)), timestamp=time.time(), wait_ack=False)
                    print ("[Telemetry] sending search frame")
                else:
                    Telemetry._link.send_object(TelemetryPackage(id, data), timestamp=time.time(), wait_ack=False)
                    print ("[Telemetry] sending log data")
                time.sleep(0.05)
                c -= 1
            
            time.sleep(0.001)

    def terminate():        
        Telemetry._run = False

class TelemetryReader:
    _link: Datalink
    _run: bool

    def __init__(self, host: str, port: int):
        self._link = Datalink(host=host, port=port, timeout=100, max_incommming_messages_in_queue=10000)
        self._run = True
        self._receive_thr = threading.Thread(target=self._receive_log_handler, daemon=True)
        self._receive_thr.start()

    def __del__(self):
        self._run = False

    def _receive_log_handler(self):
        while self._run:
            if not self._link.is_ready():
                time.sleep(0.001)
                continue

            if not self._link.has_data():
                time.sleep(0.001)
                continue

            print ("receiving object")

            inc, timestamp = self._link.recv_object()
            if inc is None:
                print ("received none object")
                continue

            self. _on_log_received(inc.id, inc.data, timestamp)
        
        del self._link

    def _on_log_received (self, id: str, data: any, timestamp: float) -> None:
        pass
            