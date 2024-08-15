import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math

from model.waypoint import Waypoint

class TestWaypoint(unittest.TestCase):

    def test_waypoint_encode_decode(self):
        p1 = Waypoint(10, 11, 20.34)
        s = str(p1)
        p2 = Waypoint.from_str(s)
        
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.z, p2.z)
        self.assertEqual(p1.heading, p2.heading)

    def test_waypoint_distance_between(self):
        p1 = Waypoint(0, 0, 0)
        p2 = Waypoint(10, 10, 0)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 10 * math.sqrt(2), 4)
        p1 = Waypoint(10, 10, 0)
        p2 = Waypoint(10, 10, 0)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 0, 4)
        p1 = Waypoint(-10, -10, 0)
        p2 = Waypoint(10, 10, 0)
        self.assertAlmostEqual(Waypoint.distance_between(p1, p2), 20 * math.sqrt(2), 4)

    def test_waypoint_compute_heading(self):
        p1 = Waypoint(0, 0, 0)
        p2 = Waypoint(0, 0, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 0, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(0, 0, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), -45, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(2, 0, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 45, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(2, 1, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 90, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(0, 1, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), -90, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(2, 2, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 135, 4)
        p1 = Waypoint(1, 1, 0)
        p2 = Waypoint(0, 2, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 225, 4)
        p1 = Waypoint(0, 0, 0)
        p2 = Waypoint(0, 4, 0)
        self.assertAlmostEqual(Waypoint.compute_heading(p1, p2), 180, 4)



if __name__ == "__main__":
    unittest.main()
