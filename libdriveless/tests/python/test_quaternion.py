import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math
from pydriveless import quaternion
from pydriveless import angle

class TestQuaternion(unittest.TestCase):

    def test_basic_properties(self):
        x = quaternion (0, 1, 0, 0)
        y = quaternion (0, 0, 1, 0)
        z = quaternion (0, 0, 0, 1)
        
        self.assertEqual(x * y, z)
        self.assertEqual(y * z, x)    
        self.assertEqual(z * x, y)
        self.assertEqual(y * x, z * -1)
        self.assertEqual(z * y, x * -1)    
        self.assertEqual(x * z, y * -1)
        
        self.assertEqual(x * 2, quaternion(0, 2, 0, 0))
        self.assertEqual(x + 2, quaternion(2, 1, 0, 0))
        self.assertEqual(quaternion(), quaternion(1, 0, 0, 0))

    def test_yaw_rotation(self):
        q1 = quaternion(0, 1, 0, 0)

        # rotating the x axis around Z
        for i in range(1, 360):
            q1.rotate_yaw(angle.new_deg(1))
            self.assertEqual(q1.yaw(), angle.new_deg(i))

        q2 = quaternion(0, 1, 0, 0)
        # rotating the x axis around Z backwards
        for i in range(359, -1, -1):
            q2.rotate_yaw(angle.new_deg(-1))
            self.assertEqual(q2.yaw(), angle.new_deg(i))

        q3 = quaternion(0, 1, 0, 0)
        q3.rotate_yaw(angle.new_deg(90))
        self.assertEqual(q3, quaternion(0, 0, 1, 0))

        q4 = quaternion(0, 1, 0, 0)
        q4.rotate_yaw(angle.new_deg(-90))
        self.assertEqual(q4, quaternion(0, 0, -1, 0))

    def test_pitch_rotation(self):
        q1 = quaternion(0, 1, 0, 0)
        q1.rotate_pitch(angle.new_deg(90))
        self.assertEqual(q1, quaternion(0, 0, 0, -1))
        a1 = q1.pitch()
        self.assertEqual(a1, angle.new_deg(90))

        q2 = quaternion(0, 1, 0, 0)
        q2.rotate_pitch(angle.new_deg(-90))
        self.assertEqual(q2, quaternion(0, 0, 0, 1))

        q3 = quaternion(0, 1, 0, 0)
        # rotating the x axis around Y
        for i in range(1, 360):
            q3.rotate_pitch(angle.new_deg(1))
            self.assertEqual(q3.pitch(), angle.new_deg(i))

        q4 = quaternion(0, 1, 0, 0)
        # rotating the x axis around Z backwards
        for i in range(359, -1, -1):
            q4.rotate_pitch(angle.new_deg(-1))
            self.assertEqual(q4.pitch(), angle.new_deg(i))

    def test_roll_rotation(self):
        q1 = quaternion(0, 0, 1, 0)
        q1.rotate_roll(angle.new_deg(90))
        self.assertEqual(q1, quaternion(0, 0, 0, 1))
        a1 = q1.roll()
        self.assertEqual(a1, angle.new_deg(90))

        q2 = quaternion(0, 0, 1, 0)
        q2.rotate_roll(angle.new_deg(-90))
        self.assertEqual(q2, quaternion(0, 0, 0, -1))
        a2 = q2.roll()
        self.assertEqual(a2, angle.new_deg(270))

        q3 = quaternion(0, 0, 1, 0)
        # rotating the x axis around Y
        for i in range(1, 360):
            q3.rotate_roll(angle.new_deg(1))
            self.assertEqual(q3.roll(), angle.new_deg(i))

        q4 = quaternion(0, 0, 1, 0)
        # rotating the x axis around Z backwards
        for i in range(359, -1, -1):
            q4.rotate_roll(angle.new_deg(-1))
            self.assertEqual(q4.roll(), angle.new_deg(i))


if __name__ == "__main__":
    unittest.main()
