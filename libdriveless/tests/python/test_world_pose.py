import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math
from pydriveless import angle
from pydriveless import WorldPose

class TestWorldPose(unittest.TestCase):

    def test_worldpose_encode_decode(self):
        p1 = WorldPose(angle.new_deg(10.1), angle.new_deg(10.2), 10.3, angle.new_deg(10.4))
        s = str(p1)
        p2 = WorldPose.from_str(s)
        
        self.assertTrue(p1.lat == p2.lat)
        self.assertTrue(p1.lon == p2.lon)
        self.assertEqual(p1.alt, p2.alt)
        self.assertTrue(p1.compass == p2.compass)

    def __compute_diff_percent(self, d1: float, d2: float) -> float:
        return 100*(1 - d1/d2)

    def test_distance_between(self):
        p1 = WorldPose(lat=angle.new_deg(5.566896), lon=angle.new_deg(95.3672), alt=0, compass=angle.new_deg(0))
        p2 = WorldPose(lat=angle.new_deg(5.566607), lon=angle.new_deg(95.370121), alt=0, compass=angle.new_deg(0))
        dist = WorldPose.distance_between(p1, p2)
        
        dist_google = 324.93
        self.assertTrue(self.__compute_diff_percent(dist, dist_google) < 0.5)
        
        p1 = WorldPose(lat=angle.new_deg(5.566896), lon=angle.new_deg(95.3672), alt=0, compass=angle.new_deg(0))
        p2 = WorldPose(lat=angle.new_deg(5.567333), lon=angle.new_deg(95.367886), alt=0, compass=angle.new_deg(0))
        dist_google=89.86
        self.assertTrue(self.__compute_diff_percent(dist, dist_google) < 0.5)

        p1 = WorldPose(lat=angle.new_deg(-29.279371052090724), lon=angle.new_deg(-56.91723210848266), alt=0, compass=angle.new_deg(0))
        p2 = WorldPose(lat=angle.new_deg(-27.92906183475516), lon=angle.new_deg(-49.75414619790477), alt=0, compass=angle.new_deg(0))
        dist = WorldPose.distance_between(p1, p2)
        dist_google=713190
        self.assertTrue(self.__compute_diff_percent(dist, dist_google) < 0.5)

    def test_compute_heading_azimuth(self):
        p1 = WorldPose(lat=angle.new_deg(-29.279371052090724), lon=angle.new_deg(-56.91723210848266), alt=0, compass=angle.new_deg(0))
        p2 = WorldPose(lat=angle.new_deg(-27.92906183475516), lon=angle.new_deg(-49.75414619790477), alt=0, compass=angle.new_deg(0))
        a = WorldPose.compute_heading(p1, p2)
        self.assertAlmostEqual(a.rad(), math.radians(79.61), 2)


if __name__ == "__main__":
    unittest.main()
