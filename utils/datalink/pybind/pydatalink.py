import ctypes
import numpy as np
LIBNAME = "/usr/local/lib/libdatalink.so"
lib = ctypes.CDLL(LIBNAME)

lib.init_server.restype = ctypes.c_void_p
lib.init_server.argtypes = [ctypes.c_int]

lib.init_client.restype = ctypes.c_void_p
lib.init_client.argtypes = [ctypes.c_char_p, ctypes.c_int]

lib.destroy_link.restype = None
lib.destroy_link.argtypes = [ctypes.c_void_p]

lib.write_str_link.restype = ctypes.c_bool
lib.write_str_link.argtypes = [ctypes.c_void_p,
                               ctypes.c_char_p, ctypes.c_long]

lib.write_np_float_link.restype = ctypes.c_bool
lib.write_np_float_link.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(
    dtype=ctypes.c_float, ndim=1), ctypes.c_size_t]

lib.write_np_uint8_link.restype = ctypes.c_bool
lib.write_np_uint8_link.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(
    dtype=ctypes.c_uint8, ndim=1), ctypes.c_size_t]


lib.has_data.restype = ctypes.c_bool
lib.has_data.argtypes = [ctypes.c_void_p]

lib.is_connected.restype = ctypes.c_bool
lib.is_connected.argtypes = [ctypes.c_void_p]

lib.is_listening.restype = ctypes.c_bool
lib.is_listening.argtypes = [ctypes.c_void_p]

lib.wait_ready.restype = ctypes.c_bool
lib.wait_ready.argtypes = [ctypes.c_void_p, ctypes.c_int]

# ctypes.POINTER(ctypes.c_char_p)
lib.read_str_link.restype = ctypes.POINTER(ctypes.c_char_p)
lib.read_str_link.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int]

# lib.read_np_link.restype = ctypes.POINTER(ctypes.c_float)
# lib.read_np_link.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]

lib.free_memory.argtypes = [ctypes.c_void_p]
lib.free_memory.restype = None

lib.terminate.argtypes = [ctypes.c_void_p]
lib.terminate.restype = None

FLOAT_SIZE_BYTES = 4  # sizeof_float


class PyDatalink:
    ''' Creates a TCP link between two endpoints
    '''
    link: ctypes.c_void_p
    timeout_ms: int

    def __init__(self, host: str = None, port: int = -1, timeout_ms: int = -1) -> None:
        if port <= 0:
            raise Exception("port should be defined")

        self.timeout_ms = timeout_ms
        self.link = None

        if host is None:
            self.link = lib.init_server(
                ctypes.c_int(port), ctypes.c_int(timeout_ms))
        else:
            self.link = lib.init_client(ctypes.c_char_p(
                host.encode('utf-8')), ctypes.c_int(port), ctypes.c_int(timeout_ms))

    def __del__(self):
        if self.link is None:
            return
        lib.destroy_link(self.link)

    def write(self, data: str) -> bool:
        return lib.write_str_link(
            self.link,
            ctypes.c_char_p(data.encode('utf-8')),
            ctypes.c_long(len(data)))

    def write_float_np(self, data: np.ndarray) -> bool:

        size = 1
        for i in range(len(data.shape)):
            size = size * data.shape[i]

        return lib.write_np_float_link(
            self.link,
            data.reshape((size)),
            size)
    def write_uint8_np(self, data: np.ndarray) -> bool:

        size = 1
        for i in range(len(data.shape)):
            size = size * data.shape[i]

        return lib.write_np_uint8_link(
            self.link,
            data.reshape((size)),
            size)
    
    def has_data(self) -> bool:
        return lib.has_data(self.link)

    def is_connected(self) -> bool:
        return lib.is_connected(self.link)

    def is_listening(self) -> bool:
        return lib.is_listening(self.link)

    def wait_ready(self) -> bool:
        return lib.wait_ready(self.link, ctypes.c_int(self.timeout_ms))
    
    def read(self) -> tuple[str, int]:
        size = ctypes.c_long(0)
        result_ptr = lib.read_str_link(self.link, ctypes.byref(size), ctypes.c_int(self.timeout_ms))
        data: str = None
        try:
            data = ctypes.cast(
                result_ptr, ctypes.c_char_p).value.decode('utf-8')
        except:
            data = ""

        lib.free(result_ptr)
        return data, size.value

    def read_float_np(self, shape: tuple) -> np.ndarray:
        size = ctypes.c_long(0)
        result_ptr = lib.read_str_link(self.link, ctypes.byref(size), ctypes.c_int(self.timeout_ms))
        data = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_char))

        data_size = int(size.value / 4)
        res = np.zeros((data_size), dtype=np.float32)

        for i in range(0, data_size):
            res[i] = ctypes.c_float.from_buffer_copy(
                data[i * FLOAT_SIZE_BYTES:i * FLOAT_SIZE_BYTES + 4]).value

        lib.free(result_ptr)
        return res.reshape(shape)
    
    def read_flatten_float_np(self) -> np.ndarray:
        size = ctypes.c_long(0)
        result_ptr = lib.read_str_link(self.link, ctypes.byref(size), ctypes.c_int(self.timeout_ms))
        data = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_char))

        data_size = int(size.value / 4)
        res = np.zeros((data_size), dtype=np.float32)

        for i in range(0, data_size):
            res[i] = ctypes.c_float.from_buffer_copy(
                data[i * FLOAT_SIZE_BYTES:i * FLOAT_SIZE_BYTES + 4]).value

        lib.free(result_ptr)
        return res

    def read_uint8_np(self, shape: tuple) -> np.ndarray:
        size = ctypes.c_long(0)
        result_ptr = lib.read_str_link(self.link, ctypes.byref(size), ctypes.c_int(self.timeout_ms))
      
        data = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_char))

        data_size = size.value

        if data_size == 0:
            lib.free(result_ptr)
            return None

        res = np.zeros((data_size), dtype=np.uint8)

        for i in range(0, data_size):
            res[i] = ctypes.c_uint8.from_buffer_copy(data[i]).value

        return res.reshape(shape)

    def terminate(self):
        lib.terminate(self.link)
