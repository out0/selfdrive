import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math

from pydriveless import WorldPose
from pydriveless import MapPose
from pydriveless import Waypoint
from pydriveless import CoordinateConverter
from pydriveless import angle

origin = WorldPose(angle.new_rad(-4.256008878655848e-09),
                     angle.new_rad(-1.5864868596990013e-08),
                     1.0150023698806763,
                     angle.new_deg(89.999237523291130))

OG_WIDTH = 600
OG_HEIGHT = 600
OG_REAL_WIDTH = 60
OG_REAL_HEIGHT = 60

class TestCoordinateConverterMapWaypoint(unittest.TestCase):



    def test_convert_map_to_waypoint_H_zero__0deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        x, y = (30, 0)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, 0)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (0, 30)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (0, -30)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_map_to_waypoint_H_zero__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        x, y = (30, 30)
        wx, wz = (600, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, 30)
        wx, wz = (600, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (30, -30)
        wx, wz = (0, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, -30)
        wx, wz = (0, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_map_to_waypoint_H_45deg__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, -15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)
 
    def test_convert_map_to_waypoint_H_minus45deg__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, 15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (15*rt, -15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_to_MapPose_H_zero__0deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        x, y = (30, 0)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, 0)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (0, 30)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (0, -30)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_to_MapPose_H_zero__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        x, y = (30, 30)
        wx, wz = (600, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, 30)
        wx, wz = (600, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (30, -30)
        wx, wz = (0, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, -30)
        wx, wz = (0, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(0)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_to_MapPose_H_45deg__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, -15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)
    
    def test_convert_to_MapPose_H_minus45deg__45deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, 15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (15*rt, -15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert(MapPose(0, 0,  0, angle.new_deg(-45)), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_H_45deg__0deg(self):
        map_coordinator = CoordinateConverter(origin, 
                                              OG_WIDTH,
                                              OG_HEIGHT,
                                              OG_REAL_WIDTH,
                                              OG_REAL_HEIGHT)
        
        rt = math.sqrt(2)

        x, y = (30*rt, 0)
        wx, wz = (600, 0)
        p = map_coordinator.convert(MapPose(0, 0, 0, angle.new_deg(-45)), MapPose(x, y, 0, angle.new_deg(0)))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        # back
        p = map_coordinator.convert(MapPose(0, 0, 0, angle.new_deg(-45)), Waypoint(wx, wz))
        self.assertAlmostEqual(p.x, x, places=4)
        self.assertAlmostEqual(p.y, y, places=4)

if __name__ == "__main__":
    unittest.main()

