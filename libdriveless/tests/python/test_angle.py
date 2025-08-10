import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math
from pydriveless import angle
from test_utils import AssertTolerance

class TestAngle(unittest.TestCase):

    def test_boolean_operators(self):
        a = angle.new_deg(12.2)
        v = angle.new_deg(12.2)
        
        self.assertTrue(a == v)
        self.assertTrue(a <= v)
        self.assertTrue(a >= v)
        
        v.set_deg(12.4)
        self.assertFalse(a == v)
        self.assertTrue(a != v)
        self.assertTrue(a < v)
        self.assertTrue(a <= v)
        self.assertTrue(v > a)
        self.assertTrue(v >= a)

    def test_compound_operators(self):
        a = angle.new_deg(12.2)
        v = angle.new_deg(12.3)
        
        c = a + v
        self.assertAlmostEqual(c.deg(), 24.5, places=4)
        
        c = a - v
        self.assertAlmostEqual(c.deg(), -0.1, places=4)
        
        self.assertAlmostEqual((a / 2).deg(), 6.1, places=4)
        self.assertTrue(angle.new_deg(6.1) == (a / 2))

    def test_conversion(self):
        a = 0.0
        while a < 360:
            b = angle.new_deg(a)
            a_rad = (a * math.pi) / 180
            
            AssertTolerance.assertAlmostEqual(self, a_rad, b.rad())
            AssertTolerance.assertAlmostEqual(self, a, b.deg())
            a += 0.0001

if __name__ == '__main__':
    unittest.main()