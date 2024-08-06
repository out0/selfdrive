import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from utils.quaternion import Quaternion


class TestQuaternion(unittest.TestCase):

    def init_q(self, w) -> np.ndarray:
        axis_angle = np.array(w)
        norm = np.linalg.norm(w)
        
        w = np.cos(norm / 2)
        if norm < 1e-50:  # to avoid instabilities and nans
            x = 0
            y = 0
            z = 0
        else:
            imag = axis_angle / norm * np.sin(norm / 2)
            x = imag[0].item()
            y = imag[1].item()
            z = imag[2].item()
        
        return np.ndarray([w, x, y, z])

    def skew_symmetric(self, v):
        """Skew symmetric form of a 3x1 vector."""
        v = v.reshape(3)
        return np.array(
            [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]], dtype=np.float64)

    def skewmult(self, q):

#        q = self.init_q(w)

        qw = q[0]
        qv = q[1:]

        om1 = np.zeros((4, 4))
        om1[0, 1:] = -qv
        om1[1:, 0] = qv
        om1[1:, 1:] = -self.skew_symmetric(qv)
        om = np.identity(4) * qw + om1

        return om

    def test_compare_qmult_skewmult (self):
       
        a = math.radians(45/2)
        c = math.cos(a)
        s = math.sin(a)

        q1 = np.array([c, 0, 0, s])
        q2 =  np.array([0, 10, 10, 0])

        mat = self.skewmult(q1)
        mat2 = self.skewmult(q2)
        res = mat @ mat2  @ np.array([c, 0, 0, -s])
        print(res)
        print("\n\n")
        mat = Quaternion.build_mult_matrix(Quaternion(w=q1[0], x=q1[1], y=q1[2],z=q1[3]))
        mat2 = Quaternion.build_mult_matrix(Quaternion(0, 10, 10, 0))
        res = mat @ mat2  @ np.array([c, 0, 0, -s])
        print (f"{res}")

if __name__ == "__main__":
    unittest.main()