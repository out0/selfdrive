import numpy as np
import math

class Quaternion(object):
    w: float
    x: float
    y: float
    z: float
    
    def __init__ (self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    
    @classmethod
    def angle_normalize(cls, a):
        """Normalize angles to lie in range -pi < a[i] <= pi."""
        a = np.remainder(a, 2*np.pi)
        a[a <= -np.pi] += 2*np.pi
        a[a  >  np.pi] -= 2*np.pi
        return a


    @classmethod
    def build_from_angles(cls, angles: np.ndarray) -> 'Quaternion':
        a = Quaternion.angle_normalize(angles)
        norm = np.linalg.norm(a)
        w = np.cos(norm / 2)
        if norm < 1e-50:  # to avoid instabilities and nans
            x = 0
            y = 0
            z = 0
        else:
            imag = a / norm * np.sin(norm / 2)
            x = imag[0].item()
            y = imag[1].item()
            z = imag[2].item()
        return Quaternion(w, x, y, z)

    @classmethod
    def build_from_vector(cls, arr: np.ndarray) -> 'Quaternion':
        return Quaternion(arr[0], arr[1], arr[2], arr[3])


    def __add__(self, other) -> 'Quaternion':
        if isinstance(other, Quaternion):
            return Quaternion (
                self.w + other.w,
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        else:
            return Quaternion (
                self.w + other,
                self.x,
                self.y,
                self.z
            )
    
    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion (
                self.w - other.w,
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        else:
            return Quaternion (
                self.w - other,
                self.x,
                self.y,
                self.z
            )
    
    @classmethod
    def build_mult_matrix(cls, q1: 'Quaternion') -> np.ndarray:
        return  np.array([
                [q1.w, -q1.x, -q1.y, -q1.z],
                [q1.x,  q1.w, -q1.z,  q1.y],
                [q1.y,  q1.z,  q1.w, -q1.x],
                [q1.z, -q1.y,  q1.x,  q1.w]])
    
    @classmethod
    def q_mul(cls, q1: 'Quaternion', q2: any) -> 'Quaternion':
        if isinstance(q2, Quaternion):
            m = Quaternion.build_mult_matrix(q1)
            
            r = np.dot(m, np.array([q2.w, q2.x, q2. y, q2.z]))
            return Quaternion(
                r[0],
                r[1],
                r[2],
                r[3],
            )            
        else:
            return Quaternion(
                q1.w * q2,
                q1.x * q2,
                q1.y * q2,
                q1.z * q2,
            )
    
    def __imul__(self, other):
        q = Quaternion.q_mul(self, other)
        self.w = q.w
        self.x = q.x
        self.y = q.y
        self.z = q.z        
        return self

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion.q_mul(self, other)
        else:
            return Quaternion(w = self.w, x = self.x * other, y = self.y * other, z = self.z * other)

    def inv(self) -> 'Quaternion':
        return self.conj() * self.size()

    def conj(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def size(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def rotate (self, x: float, y: float, z: float, angle_rad: float) -> 'Quaternion':
        a = angle_rad / 2
        c = math.cos(a)
        s = math.sin(a)
        q = Quaternion(0, x, y, z) * s + c
        return q * self * q.inv()
    
    def __truediv__(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError(f"unsupported operand type(s) for /=: 'Quaternion' and '{type(other).__name__}'")
        prod = self * other.inv()
        self.w = prod.w
        self.x = prod.x
        self.y = prod.y
        self.z = prod.z
        return self

    def __str__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

    def duplicate(self) -> 'Quaternion':
        return Quaternion(self.w, self.x, self.y, self.z)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    def get_rotation_matrix(self, angle_rad: float) -> np.ndarray:
        
        a = angle_rad / 2
        c = math.cos(a)
        s = math.sin(a)
        q = Quaternion(0, self.x, self.y, self.z) * s + c
                
        q0 = q.w
        q1 = q.x
        q2 = q.y
        q3 = q.z
         
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
         
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
         
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
         
        # 3x3 rotation matrix
        m = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])

        t = np.array([[0.0], [0.0], [0.0]])

        # 4 x 4 rotation matrix to apply on vectors ([x y z 1])
        return np.vstack([np.hstack([m, t]), np.array([0.0, 0.0, 0.0, 1.0])])
            
