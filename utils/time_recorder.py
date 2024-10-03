from queue import Queue
import time
from threading import Lock, Thread

EXECUTION_FILE_REPORT = "execution_time_results.log"

class ExecutionTimeRecorder:
    _file: str = None
    _execution_start_time: dict
    _mtx: Lock
    _file_mtx: Lock

    @classmethod
    def get_file_name(cls) -> str:
        if cls._file is None:
            return EXECUTION_FILE_REPORT
        return ExecutionTimeRecorder._file

    @classmethod
    def initialize(cls, file: str = EXECUTION_FILE_REPORT) -> None:
        ExecutionTimeRecorder._execution_data = Queue()
        ExecutionTimeRecorder._file = file
        ExecutionTimeRecorder._execution_start_time = {}
        ExecutionTimeRecorder._mtx = Lock()
        ExecutionTimeRecorder._file_mtx = Lock()

    @classmethod
    def start(cls, module: str) -> None:       
        ExecutionTimeRecorder._mtx.acquire(blocking=True)        
        ExecutionTimeRecorder._execution_start_time[module] = time.time()
        ExecutionTimeRecorder._mtx.release()

    @classmethod
    def stop(cls, module: str) -> None:
        if ExecutionTimeRecorder._execution_start_time is None:
            return
        t2 = time.time() 
        ExecutionTimeRecorder._mtx.acquire(blocking=True)
        t1 = 0.0 + ExecutionTimeRecorder._execution_start_time[module]
        #ExecutionTimeRecorder._execution_start_time[module] = None
        ExecutionTimeRecorder._mtx.release()      
        ExecutionTimeRecorder.update_file(module, 1000 * (t2 - t1))

    @classmethod
    def update_file(cls, module: str, elapsed_time_ms: float) -> None:
        thr = Thread(target=ExecutionTimeRecorder.update_file_from_queue, args=(ExecutionTimeRecorder._file, module, elapsed_time_ms))
        thr.start()

    def update_file_from_queue(file: str, module: str, elapsed_time_ms: float) -> None:
        ExecutionTimeRecorder._file_mtx.acquire(blocking=True)
        with open(ExecutionTimeRecorder._file, "a") as f:
            f.write(f"{module}:{elapsed_time_ms}\n")
        ExecutionTimeRecorder._file_mtx.release()



