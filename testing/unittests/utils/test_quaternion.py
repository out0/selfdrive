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
    
    def test_pure_quat_rotation(self):
        q2 = Quaternion(0, 10, 10, 0)
        # + 45 degrees Z rotation on (10,10) should result in (0, 10*sqrt(2))
        qr = q2.rotate(0, 0, 1, math.radians(45))
        self.assertAlmostEqual(10*math.sqrt(2), qr.y, places=4)
        self.assertAlmostEqual(0, qr.x, places=4)

        # - 45 degrees Z rotation on (10,10) should result in (10*sqrt(2), 0)
        qr = q2.rotate(0, 0, 1, math.radians(-45))
        self.assertAlmostEqual(10*math.sqrt(2), qr.x, places=4)
        self.assertAlmostEqual(0, qr.y, places=4)
        

        # + 45 degrees -Z rotation on (10,10) should result in (10*sqrt(2), 0) because the Z axis is now inverted!
        qr = q2.rotate(0, 0, -1, math.radians(45))
        self.assertAlmostEqual(10*math.sqrt(2), qr.x, places=4)
        self.assertAlmostEqual(0, qr.y, places=4)



    def test_quat_rotation_using_matrices(self):
        
        unit_axis_quaternion = Quaternion(0, 0, 0, 1)  # z
        q = Quaternion(0, 10, 10, 0)  #  (x = 10, y = 10)  +45 degree vector
        
        ## quarternion rotation of q is performed by the following:
        #
        # let qrot = quaternion for the rotation operation
        # 
        # qrot is obtained by w = cos(angle/2) + sin(angle/2) * unit_axis_quaternion
        # 
        # q_result = qrot * q * inv(qrot)
        #
        
        a = math.radians(45/2)
        qrot = unit_axis_quaternion * math.sin(a) + math.cos(a)
                
        m1 = Quaternion.build_mult_matrix(qrot)  # m1 gives a 4x4 matrix that can be used to multiply qrot by another quaternion
        m2 = Quaternion.build_mult_matrix(q)
        
        res = m1 @ m2 @ qrot.inv().to_matrix()
        
        self.assertAlmostEqual(10*math.sqrt(2), res[2], places=4)
        self.assertAlmostEqual(0, res[1], places=4)
        
        # full step-by-step transformation:
        
        q = Quaternion(0, 10, 10, 0)  #  (x = 10, y = 10)  +45 degree vector
        #print(f"target: {q}")
        
        qrot = unit_axis_quaternion * math.sin(a) + math.cos(a)
        #print(f"rotation axis: {qrot}")
        
        m1 = Quaternion.build_mult_matrix(qrot)
        
        rotation_first_part = m1 @ q.to_matrix()
        #print (f"first half of the quaternion rotation: {rotation_first_part}")
        rotation_second_part_quat = Quaternion.build_from_vector(rotation_first_part)
        
        m2 = Quaternion.build_mult_matrix(rotation_second_part_quat)
        
        #print(f"inverted rotation axis : {qrot.inv()}")        
        res = m2 @ qrot.inv().to_matrix()
        
        a = math.radians(45/2)
        c = math.cos(a)
        s = math.sin(a)
        q1 = np.array([c, 0, 0, s])
        q2 = np.array([0, 10, 10, 0])

        mat = self.skewmult(q1)
        mat2 = self.skewmult(q2)
        res = mat @ mat2  @ np.array([c, 0, 0, -s])
        
        
        #print (f"result: {res}")
        self.assertAlmostEqual(10*math.sqrt(2), res[2], places=4)
        self.assertAlmostEqual(0, res[1], places=4)
       

    def test_compare_qmult_skewmult (self):
       
        a = math.radians(45/2)
        c = math.cos(a)
        s = math.sin(a)

        q1 = np.array([c, 0, 0, s])
        q2 = np.array([0, 10, 10, 0])

        mat = self.skewmult(q1)
        mat2 = self.skewmult(q2)
        res = mat @ mat2  @ np.array([c, 0, 0, -s])
        print(res)
        print("\n\n")
        mat = Quaternion.build_mult_matrix(Quaternion.build_from_vector(q1))
        mat2 = Quaternion.build_mult_matrix(Quaternion(0, 10, 10, 0))
        res = mat @ mat2  @ np.array([c, 0, 0, -s])
        print (f"{res}")

if __name__ == "__main__":
    unittest.main()