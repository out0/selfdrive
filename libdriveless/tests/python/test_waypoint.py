import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math
from pydriveless import angle
from pydriveless import Waypoint

class TestWaypoint(unittest.TestCase):
    
    def test_waypoint_encode_decode(self):
        p1 = Waypoint(10, 11, angle.new_deg(20.34))
        s = str(p1)
        p2 = Waypoint.from_str(s)
        
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.z, p2.z)
        self.assertTrue(p1.heading == p2.heading)

    def test_waypoint_distance_between(self):
        p1 = Waypoint(0, 0)
        p2 = Waypoint(10, 10)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 10 * math.sqrt(2), 4)
        p1 = Waypoint(10, 10)
        p2 = Waypoint(10, 10)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 0, 4)
        p1 = Waypoint(-10, -10)
        p2 = Waypoint(10, 10)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 20 * math.sqrt(2), 4)

    def test_waypoint_compute_heading(self):
        p1 = Waypoint(0, 0)
        p2 = Waypoint(0, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 0, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(0, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), -45, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(2, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 45, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(2, 1)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 90, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(0, 1)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), -90, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(2, 2)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 135, 4)
        p1 = Waypoint(1, 1)
        p2 = Waypoint(0, 2)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 225, 4)
        p1 = Waypoint(0, 0)
        p2 = Waypoint(0, 4)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2).deg(), 180, 4)


    def test_clone (self) -> None:
        p1 = Waypoint(x=1, z=2, heading=angle.new_deg(3))
        p2 = p1.clone()
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.z, p2.z)
        self.assertEqual(p1.heading, p2.heading)
        self.assertEqual(p1, p2)
        self.assertFalse(p1 is p2)

if __name__ == "__main__":
    unittest.main()
