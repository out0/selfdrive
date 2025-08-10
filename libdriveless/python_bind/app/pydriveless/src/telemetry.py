
import threading
from collections import deque
import numpy as np
import cv2
from pydriveless import SearchFrame
import time, json

class Telemetry:
    _data_queue: deque
    _mtx: threading.Lock
    _write_log_thr: threading.Thread
    _run: bool

    def _initialize():
        if hasattr(Telemetry, "_data_queue"):
            return
        
        Telemetry._run = True
        Telemetry._data_queue = deque()
        Telemetry._mtx = threading.Lock()
        Telemetry._write_log_thr = threading.Thread(target=Telemetry.write_log_handler, daemon=True)
        Telemetry._write_log_thr.start()
    
    def log_if(condition: bool, file: str, data: any, append: bool = False):
        if not condition: return
        Telemetry.log(file, data, append)

    def log(file: str, data: any, append: bool = False):
        Telemetry._initialize()
        Telemetry._mtx.acquire()
        Telemetry._data_queue.append((file, data, append))
        Telemetry._mtx.release()

    def empty():
        Telemetry._initialize()
        Telemetry._mtx.acquire()
        empty = len(Telemetry._data_queue) == 0
        Telemetry._mtx.release()
        return empty

    def write_log_handler():
        while Telemetry._run:
            Telemetry._mtx.acquire()
            if len(Telemetry._data_queue) > 0:
                item = Telemetry._data_queue.pop()
            else:
                item = None
            Telemetry._mtx.release()

            if item is None:
                continue

            file, data, append = item
            #print (f"logging file {file} {" [append]" if append else ""}")

            if isinstance(data, np.ndarray):
                cv2.imwrite(file, data)
            
            elif isinstance(data, SearchFrame):
                f = data.get_frame()
                cv2.imwrite(file, f.astype(np.uint8))

            elif isinstance(data, list):
                res_str = []
                for p in data:
                    res_str.append(str(p))
                with open(file, "a" if append else "w") as f:
                    f.write(json.dumps(res_str))
            
            else:
                with open(file, "a" if append else "w") as f:
                    f.write(f"{str(data)}\n")

            time.sleep(0.05)

    def terminate():
        if hasattr(Telemetry, "_data_queue"):
            return
        Telemetry._run = False