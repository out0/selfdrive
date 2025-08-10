import math
import ctypes
import numpy as np
from .angle import angle
import os

class quaternion:
    __q: ctypes.c_void_p
       
    def __init__(self, w: float = 1, x: float = 0, y: float = 0, z: float = 0):
        self.__q = None
        quaternion.setup_cpp_lib()
        if (w != 0 or x != 0 or y != 0 or z != 0):
            self.__q = quaternion.lib.p__quaternion_new(w, x, y, z)
       
    def __del__(self) -> None:
        if self.__q is not None:
            quaternion.lib.p__quaternion_destroy(self.__q)
            self.__q = None
    
    
    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(quaternion, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libdriveless.so")
            
        quaternion.lib = ctypes.CDLL(lib_path)
        
        quaternion.lib.p__quaternion_new.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_new.argtypes = [
            ctypes.c_double, # w
            ctypes.c_double, # x
            ctypes.c_double, # y
            ctypes.c_double # z
        ]       
        quaternion.lib.p__quaternion_destroy.restype = None
        quaternion.lib.p__quaternion_destroy.argtypes = [
            ctypes.c_void_p
        ]
        
        
        quaternion.lib.p__quaternion_sum_scalar.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_sum_scalar.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # scalar
        ]
        quaternion.lib.p__quaternion_sum.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_sum.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        quaternion.lib.p__quaternion_minus_scalar.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_minus_scalar.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # scalar
        ]
        quaternion.lib.p__quaternion_minus.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_minus.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        
        quaternion.lib.p__quaternion_mul_scalar.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_mul_scalar.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # other
        ]
        quaternion.lib.p__quaternion_mul.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_mul.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        quaternion.lib.p__quaternion_div_scalar.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_div_scalar.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        quaternion.lib.p__quaternion_div.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_div.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        quaternion.lib.p__quaternion_equals.restype = ctypes.c_bool
        quaternion.lib.p__quaternion_equals.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_void_p  # other
        ]
        
        quaternion.lib.p__quaternion_invert.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_invert.argtypes = [
            ctypes.c_void_p, # this
        ]
        quaternion.lib.p__quaternion_conjugate.restype = ctypes.c_void_p
        quaternion.lib.p__quaternion_conjugate.argtypes = [
            ctypes.c_void_p, # this
        ]
        quaternion.lib.p__quaternion_size.restype = ctypes.c_double
        quaternion.lib.p__quaternion_size.argtypes = [
            ctypes.c_void_p, # this
        ]
        
        quaternion.lib.p__rotate_yaw.restype = None
        quaternion.lib.p__rotate_yaw.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # angle(radians)
        ]
        quaternion.lib.p__rotate_pitch.restype = None
        quaternion.lib.p__rotate_pitch.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # angle(radians)
        ]
        quaternion.lib.p__rotate_roll.restype = None
        quaternion.lib.p__rotate_roll.argtypes = [
            ctypes.c_void_p, # this
            ctypes.c_double  # angle(radians)
        ]
        
        quaternion.lib.p__yaw.restype = ctypes.c_double
        quaternion.lib.p__yaw.argtypes = [
            ctypes.c_void_p # this
        ]
        quaternion.lib.p__pitch.restype = ctypes.c_double
        quaternion.lib.p__pitch.argtypes = [
            ctypes.c_void_p # this
        ]
        quaternion.lib.p__roll.restype = ctypes.c_double
        quaternion.lib.p__roll.argtypes = [
            ctypes.c_void_p # this
        ]
        
        quaternion.lib.p__value.restype = ctypes.POINTER(ctypes.c_double)
        quaternion.lib.p__value.argtypes = [
            ctypes.c_void_p # this
        ]
        quaternion.lib.p__value_free.restype = None
        quaternion.lib.p__value_free.argtypes = [
            ctypes.POINTER(ctypes.c_double) # this
        ]
       
        
    def __add__(self, value) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)        
        if not isinstance(value, quaternion):
            q.__q = quaternion.lib.p__quaternion_sum_scalar(self.__q, value)
        else:
            q.__q = quaternion.lib.p__quaternion_sum(self.__q, value.__q)
        return q

    def __sub__(self, value) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)        
        if not isinstance(value, quaternion):
            q.__q = quaternion.lib.p__quaternion_minus_scalar(self.__q, value)
        else:
            q.__q = quaternion.lib.p__quaternion_minus(self.__q, value.__q)
        return q

    def __mul__(self, value) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)        
        if not isinstance(value, quaternion):
            q.__q = quaternion.lib.p__quaternion_mul_scalar(self.__q, value)
        else:
            q.__q = quaternion.lib.p__quaternion_mul(self.__q, value.__q)
        return q
    
    def __truediv__(self, value) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)        
        if not isinstance(value, quaternion):
            q.__q = quaternion.lib.p__quaternion_div_scalar(self.__q, value)
        else:
            q.__q = quaternion.lib.p__quaternion_div(self.__q, value.__q)
        return q

    def  __eq__(self, value):
        if not isinstance(value, quaternion):
            return False
        return quaternion.lib.p__quaternion_equals(self.__q, value.__q)
    
    def  __ne__(self, value: 'quaternion'):
        return not self.__eq__(value)
    
    def __str__(self):
        values = self.to_array()
        return f"({values[0]}, {values[1]}, {values[2]}, {values[3]})"
        

    def invert(self) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)
        q.__q = quaternion.lib.p__quaternion_invert(self.__q)
        return q
        
    def conjugate(self) -> 'quaternion':
        q = quaternion(0, 0, 0, 0)
        q.__q = quaternion.lib.p__quaternion_conjugate(self.__q)
        return q
    
    def __len__(self) -> float:
        return quaternion.lib.p__quaternion_size(self.__q)

    def rotate_yaw(self, a: angle) -> None:
        quaternion.lib.p__rotate_yaw(self.__q, a.rad())

    def rotate_pitch(self, a: angle) -> None:
        quaternion.lib.p__rotate_pitch(self.__q, a.rad())

    def rotate_roll(self, a: angle) -> None:
        quaternion.lib.p__rotate_roll(self.__q, a.rad())


    def yaw(self) -> angle:
        return angle.new_rad(quaternion.lib.p__yaw(self.__q))

    def pitch(self) -> angle:
        return angle.new_rad(quaternion.lib.p__pitch(self.__q))

    def roll(self) -> angle:
        return angle.new_rad(quaternion.lib.p__roll(self.__q))
        
    def to_array (self) -> np.ndarray:
        ptr = quaternion.lib.p__value(self.__q)
        res = np.array([ptr[0], ptr[1], ptr[2], ptr[3]])
        quaternion.lib.p__value_free(ptr)
        return res

if __name__ == "__main__":
    x = quaternion(0, 1, 0, 0)
    x.rotate_yaw(angle.new_deg(90))
    vals = x.to_array()
    print(vals)
    print (x.yaw().deg())