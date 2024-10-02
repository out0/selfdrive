from queue import Queue
import time
from threading import Lock, Thread

EXECUTION_FILE_REPORT = "execution_time_results.log"

class ExecutionTimeRecorder:
    __file: str = None
    __execution_data: Queue
    __execution_agents: dict
    __mtx: Lock
    __file_mtx: Lock

    @classmethod
    def get_file_name(cls) -> str:
        if cls.__file is None:
            return EXECUTION_FILE_REPORT
        return ExecutionTimeRecorder.__file

    @classmethod
    def initialize(cls, file: str = EXECUTION_FILE_REPORT) -> None:
        ExecutionTimeRecorder.__execution_data = Queue()
        ExecutionTimeRecorder.__file = file
        ExecutionTimeRecorder.__execution_agents = {}
        ExecutionTimeRecorder.__mtx = Lock()
        ExecutionTimeRecorder.__file_mtx = Lock()


    @classmethod
    def start(cls, module: str) -> None:
        if not hasattr(cls, '__execution_data'):
            ExecutionTimeRecorder.initialize()
        
        ExecutionTimeRecorder.__mtx.acquire(blocking=True)
        ExecutionTimeRecorder.__execution_agents[module] = time.time()
        ExecutionTimeRecorder.__mtx.release()

    @classmethod
    def stop(cls, module: str) -> None:
        if ExecutionTimeRecorder.__execution_data is None:
            ExecutionTimeRecorder.initialize()
        
        ExecutionTimeRecorder.__mtx.acquire(blocking=True)
        t1 = ExecutionTimeRecorder.__execution_agents[module]
        ExecutionTimeRecorder.__execution_agents[module] = None
        t2 = 1000 * (time.time() - t1)

        exec_item = {
            'module': module,
            'exec': t2
        }
        ExecutionTimeRecorder.__execution_data.put(exec_item)
        ExecutionTimeRecorder.__mtx.release()

        ExecutionTimeRecorder.update_file()
    
    @classmethod
    def update_file(cls) -> None:
        thr = Thread(target=ExecutionTimeRecorder.__update_file_from_queue)
        thr.start()

    @classmethod
    def __update_file_from_queue(cls) -> None:
        items = []
        ExecutionTimeRecorder.__mtx.acquire(blocking=True)
        while not ExecutionTimeRecorder.__execution_data.empty():
            item = ExecutionTimeRecorder.__execution_data.get(block=False)
            if item is not None:
                items.append(item)
        ExecutionTimeRecorder.__mtx.release()

        if len(items) > 0:
            ExecutionTimeRecorder.__file_mtx.acquire(blocking=True)

            with open(ExecutionTimeRecorder.__file, "a") as f:
                for i in items:
                    f.write(f"{i['module']}:{i['exec']}\n")
            
            ExecutionTimeRecorder.__file_mtx.release()




