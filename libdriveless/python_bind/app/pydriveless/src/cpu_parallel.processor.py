import ctypes, os

class ParallelProcessor:
    def __init__(self, num_thread_handlers: int, num_blocks: int, num_threads_per_block: int):
        self._num_thread_handlers = num_thread_handlers
        self._num_blocks = num_blocks
        self._num_threads_per_block = num_threads_per_block


    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(ParallelProcessor, "_lib"):
            return

        lib_path = os.path.join(os.path.dirname(
            __file__), "../cpp", "libdriveless.so")

        ParallelProcessor._lib = ctypes.CDLL(lib_path)
        
        ParallelProcessor._lib.run_parallel_func.argtypes = [
            ctypes.CFUNCTYPE(None, ctypes.c_int), 
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        ParallelProcessor._lib.run_parallel_func.restype = None

    def handler(self, thread_id: int):
        pass

    def run_and_wait(self):
        self._lib.run_parallel_func(
            self.handler,
            self._num_thread_handlers,
            self._num_blocks,
            self._num_threads_per_block
        )