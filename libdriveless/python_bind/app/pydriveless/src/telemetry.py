
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
        max_queued_items: int = 10,
        send_message_delay_ms: int = 50,
        enable: bool = True
    ):
        Telemetry._send_msg_delay = send_message_delay_ms/1000

        if enable:
            Telemetry._link = Datalink(port=port, timeout=1000, max_incommming_messages_in_queue=1)
            Telemetry._max_items = max_queued_items
            Telemetry._run = True
            Telemetry._data_queue = deque()
            Telemetry._write_log_thr = threading.Thread(target=Telemetry.write_log_handler, daemon=True)
            Telemetry._write_log_thr.start()            
        else:
            Telemetry._run = False

        if Telemetry._send_msg_delay <= 0:
            Telemetry._send_msg_delay = 0.05

    def log_if(condition: bool, id: str, data: any):
        if not condition: return
        Telemetry.log(id, data)

    def log(id: str, data: any):
        if Telemetry._run:
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
                print ("[Telemetry] waiting for the client to connect")
                while not Telemetry._link.is_ready():
                    time.sleep(0.001)
                    continue
                print ("[Telemetry] client connected")

            c = len(Telemetry._data_queue)
            while c > 0:
                item = Telemetry._data_queue.popleft()
                if item is None: 
                    #print ("[Telemetry] null queued item")
                    continue   
                id, data = item

                if isinstance(data, SearchFrame):
                    f = data.get_frame()
                    Telemetry._link.send_object(TelemetryPackage(id, f.astype(np.uint8)), timestamp=time.time(), wait_ack=False)
                    #print ("[Telemetry] sending search frame")
                elif isinstance(data, np.ndarray):
                    raw_data = data.tobytes()
                    Telemetry._link.send_object(TelemetryPackage(id, raw_data), timestamp=time.time(), wait_ack=False)
                else:
                    Telemetry._link.send_object(TelemetryPackage(id, data), timestamp=time.time(), wait_ack=False)
                    #print ("[Telemetry] sending log data")
                time.sleep(Telemetry._send_msg_delay)
                c -= 1
            
            time.sleep(0.001)

    def terminate():        
        Telemetry._run = False
        Telemetry._write_log_thr.join()
        Telemetry._data_queue.clear()
        del Telemetry._link
        Telemetry._link = None

class TelemetryReader:
    _link: Datalink
    _run: bool

    def __init__(self, host: str, port: int):
        self._link = Datalink(host=host, port=port, timeout=1000, max_incommming_messages_in_queue=10000)
        self._run = True
        self._receive_thr = threading.Thread(target=self._receive_log_handler, daemon=True)
        self._receive_thr.start()

    def __del__(self):
        self._run = False

    def _receive_log_handler(self):
        while self._run:

            if not self._link.is_ready():
                print ("[client] not connected")
                while not self._link.is_ready():
                    time.sleep(0.001)
                print ("client] connected")
                continue

            if not self._link.has_data():
                time.sleep(0.001)
                continue

            #print ("receiving object")

            inc, timestamp = self._link.recv_object()
            if inc is None:
                #print ("received none object")
                continue

            self. _on_log_received(inc.id, inc.data, timestamp)
            self._link.write_keep_alive()
        
        del self._link

    def _on_log_received (self, id: str, data: any, timestamp: float) -> None:
        pass
            