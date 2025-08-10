
PI = 3.1415926535897931e+0
HALF_PI = 1.5707963267948966e+0
QUARTER_PI = 0.7853981633974483e+0
DOUBLE_PI = 6.2831853071795862e+0
TO_RAD = 0.017453293
TO_DEG = 57.295779513
EQUALITY_TOLERANCE = 0.001

Q3_INIT = 1.5 * PI

def _TOLERANCE_EQUALITY(a, b) -> bool:
    return abs(a - b) < EQUALITY_TOLERANCE
def _TOLERANCE_LOWER(a, b):
    return a - b < EQUALITY_TOLERANCE
def _TOLERANCE_GREATER(a, b):
    return a - b > EQUALITY_TOLERANCE

class angle:
    __val_rad: float
    
    def __init__(self, val_rad: float):
        self.__val_rad = val_rad
    
    @classmethod
    def new_rad(cls, val: float) -> 'angle':
        return angle(val)
    
    @classmethod
    def new_deg(cls, val: float) -> 'angle':
        return angle(val * TO_RAD)

    def rad(self) -> float:
        return self.__val_rad
    
    def deg(self) -> float:
        return TO_DEG * self.__val_rad
       
    def __add__(self, value) -> 'angle':
        if not isinstance(value, angle):
            return angle.new_rad(self.rad() + value)
        return angle.new_rad(self.rad() + value.rad())

    def __sub__(self, value) -> 'angle':
        if not isinstance(value, angle):
            return angle.new_rad(self.rad() - value)
        return angle.new_rad(self.rad() - value.rad())


    def __mul__(self, value) -> 'angle':
        if not isinstance(value, angle):
            return angle.new_rad(self.rad() * value)
        return angle.new_rad(self.rad() * value.rad())
    
    
    def __truediv__(self, value) -> 'angle':
        if not isinstance(value, angle):
            return angle.new_rad(self.rad() / value)
        return angle.new_rad(self.rad() / value.rad())

    def  __eq__(self, value):
        if not isinstance(value, angle):
            return False
        
        v1 = self.rad()
        v2 = value.rad()
        
        if (v2 >= 0 and v2 < HALF_PI and v1 > Q3_INIT and v1 < DOUBLE_PI):
            return _TOLERANCE_EQUALITY(abs(v1 - DOUBLE_PI), v2)
        if (v1 >= 0 and v1 < HALF_PI and v2 > Q3_INIT and v2 < DOUBLE_PI):
            return _TOLERANCE_EQUALITY(abs(v2 - DOUBLE_PI), v1)
        return _TOLERANCE_EQUALITY(self.rad(), value.rad())
    
    def  __ne__(self, value: 'angle'):
        return not self.__eq__(value)
    
    def  __lt__(self, value: 'angle'):
        if not isinstance(value, angle):
            return _TOLERANCE_LOWER(self.rad(), value)
        return _TOLERANCE_LOWER(self.rad(), value.rad())
        
    def  __gt__(self, value: 'angle'):
        if not isinstance(value, angle):
            return _TOLERANCE_GREATER(self.rad(), value)
        return _TOLERANCE_GREATER(self.rad(), value.rad())
    
        
    def  __le__(self, value: 'angle'):
        if not isinstance(value, angle):
            return self.__eq__(value) or _TOLERANCE_LOWER(self.rad(), value)
        return self.__eq__(value) or _TOLERANCE_LOWER(self.rad(), value.rad())
        
    def  __ge__(self, value: 'angle'):
        if not isinstance(value, angle):
            return self.__eq__(value) or _TOLERANCE_GREATER(self.rad(), value)
        return self.__eq__(value) or _TOLERANCE_GREATER(self.rad(), value.rad())
    
    def set_deg(self, val: float):
       self.__val_rad = TO_RAD * val

    def set_rad(self, val: float):
       self.__val_rad = val
       
    

    