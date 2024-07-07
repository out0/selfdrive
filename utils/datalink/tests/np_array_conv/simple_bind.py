import ctypes
import numpy as np
import struct

LIBNAME = "./libsimplelib.so"
lib = ctypes.CDLL(LIBNAME)

lib.py_gen_float_data.restype = ctypes.POINTER(ctypes.c_char_p)
lib.py_gen_float_data.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_int)]

lib.py_free_mem.restype = None
lib.py_free_mem.argtypes = [ctypes.c_void_p]

lib.py_simulate_send_float_array.restype = None
lib.py_simulate_send_float_array.argtypes = [np.ctypeslib.ndpointer(
    dtype=ctypes.c_float, ndim=1), ctypes.c_size_t]

def gen_float_array(size: int) -> np.ndarray:
    word_size = ctypes.c_int(0)
    result_ptr = lib.py_gen_float_data(ctypes.c_long(size), ctypes.byref(word_size))  # char *p
    data = ctypes.cast(result_ptr,ctypes.POINTER(ctypes.c_char))
    res = np.zeros((size), dtype=np.float32)

    s = word_size.value

    for i in range(0, size):
        val = ctypes.c_float.from_buffer_copy(data[i * s:i * s + 4]).value
        res[i] = val

    lib.py_free_mem(result_ptr)
    return res

def simulate_send_float_array(data: np.ndarray) -> None:
    size = 1
    for i in range(len(data.shape)):
        size = size * data.shape[i]

    lib.py_simulate_send_float_array(
        data.reshape((size)),
        (size))

if __name__ == "__main__":
    arr = gen_float_array(100)
    simulate_send_float_array(arr)
