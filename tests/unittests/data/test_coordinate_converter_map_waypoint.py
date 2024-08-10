import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math

from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.waypoint import Waypoint
from data.coordinate_converter import CoordinateConverter
from model.physical_parameters import PhysicalParameters


class TestCoordinateConverterMapWaypoint(unittest.TestCase):

    def set_physical_params(self) -> None:
        PhysicalParameters.OG_WIDTH = 600
        PhysicalParameters.OG_HEIGHT = 600
        PhysicalParameters.OG_REAL_WIDTH = 60
        PhysicalParameters.OG_REAL_HEIGHT = 60


    def test_convert_map_to_waypoint_H_zero__0deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        x, y = (30, 0)
        wx, wz = (300, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, 0)
        wx, wz = (300, 600)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (0, 30)
        wx, wz = (600, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (0, -30)
        wx, wz = (0, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_map_to_waypoint_H_zero__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        x, y = (30, 30)
        wx, wz = (600, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, 30)
        wx, wz = (600, 600)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (30, -30)
        wx, wz = (0, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-30, -30)
        wx, wz = (0, 600)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0, 0, 0), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_map_to_waypoint_H_45deg__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, 45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, 45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, 45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, -15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, 45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)
 
    def test_convert_map_to_waypoint_H_minus45deg__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, -45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, 15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, -45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (15*rt, -15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, -45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        x, y = (-15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, -45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

    def test_convert_to_MapPose_H_zero__0deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        x, y = (30, 0)
        wx, wz = (300, 0)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, 0)
        wx, wz = (300, 600)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (0, 30)
        wx, wz = (600, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (0, -30)
        wx, wz = (0, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_to_MapPose_H_zero__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        x, y = (30, 30)
        wx, wz = (600, 0)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, 30)
        wx, wz = (600, 600)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (30, -30)
        wx, wz = (0, 0)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-30, -30)
        wx, wz = (0, 600)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0, 0, 0), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_to_MapPose_H_45deg__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, 45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, 45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, 45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, -15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, 45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)
    
    def test_convert_to_MapPose_H_minus45deg__45deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        rt = math.sqrt(2)

        x, y = (15*rt, 15*rt)
        wx, wz = (600, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, -45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, 15*rt)
        wx, wz = (300, 600)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, -45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (15*rt, -15*rt)
        wx, wz = (300, 0)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, -45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

        x, y = (-15*rt, -15*rt)
        wx, wz = (0, 300)
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, -45), Waypoint(wx, wz))
        self.assertAlmostEqual(x, p.x, places=4)
        self.assertAlmostEqual(y, p.y, places=4)

    def test_convert_H_45deg__0deg(self):
        self.set_physical_params()
        map_coordinator = CoordinateConverter(WorldPose(0, 0, 0, 0))
        
        rt = math.sqrt(2)

        x, y = (30*rt, 0)
        wx, wz = (600, 0)
        p = map_coordinator.convert_map_to_waypoint(MapPose(0, 0,  0, -45), MapPose(x, y, 0, 0))
        self.assertEqual(wx, p.x)
        self.assertEqual(wz, p.z)

        # back
        p = map_coordinator.convert_waypoint_to_map_pose(MapPose(0, 0,  0, -45), Waypoint(wx, wz))
        self.assertAlmostEqual(p.x, x, places=4)
        self.assertAlmostEqual(p.y, y, places=4)

if __name__ == "__main__":
    unittest.main()

