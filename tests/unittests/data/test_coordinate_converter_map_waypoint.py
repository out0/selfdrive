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

    def test_convert_map_to_waypoint_H_zero__0deg(self):
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
    PhysicalParameters.OG_WIDTH = 600
    PhysicalParameters.OG_HEIGHT = 600
    PhysicalParameters.OG_REAL_WIDTH = 60
    PhysicalParameters.OG_REAL_HEIGHT = 60
    unittest.main()



#
# Data built with the following script
#
# ---------------------------------------------

# import sys, os
# sys.path.append("../../")
# from carlasim.carla_client import CarlaClient
# from carlasim.carla_ego_car import CarlaEgoCar
# from carlasim.sensors.data_sensors import *

# client = CarlaClient(town='Town07')
# ego = CarlaEgoCar(client)
# ego.init_fake_bev_seg_camera()
# ego.set_pose(0, 0, 0, 0)
# ego.set_brake(1.0)

# import time
# from model.map_pose import MapPose
# from data.coordinate_converter import CoordinateConverter
# from model.world_pose import WorldPose

# ego.set_pose(0, 0, 0, 0)
# time.sleep(1)
# ego.set_brake(1.0)
# d = ego.get_gps().read()

# initial_world_pose = WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)
# p = ego.get_gps().get_location()
# map_carla = MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2))

# print(f"GPS: {d.latitude}, {d.longitude}, {d.altitude}")
# print(f"Base World Pose: {initial_world_pose}")
# print(f"Location in Carla: {map_carla}")

# conv = CoordinateConverter(initial_world_pose)
# map_pose1 = conv.convert_to_map_pose(initial_world_pose)
# print(f"Initial map pose: {map_pose1}")

# ego.set_pose(-3, -22, 1, 90)
# time.sleep(1)
# ego.set_brake(1.0)
# d = ego.get_gps().read()
# time.sleep(1)
# wp2 = WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)
# p = ego.get_gps().get_location()
# map_truth = MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2))
# print(f"GPS: {d.latitude}, {d.longitude}, {d.altitude}")
# print(f"World Pose: {wp2}")
# print(f"Location in Carla: {map_truth}")
# converted_map = conv.convert_to_map_pose(wp2)
# print(converted_map)
# print(f"Converted map: {converted_map}")

# poses = [
#     (0, 0, 1, 0),
#     (0, 0, 1, -90),
#     (0, 0, 1, -135),
#     (-4, -4, 1, -135),
#     (-4, -20, 1, -90),
#     (-4, 20, 1, 90),
#     (5, 60, 1, 0),
#     (5, 60, 1, 90),
#     (5, 60, 1, 135),
# ]

# for i in range (len(poses)):
#     ego.set_pose(poses[i][0], poses[i][1], poses[i][2], poses[i][3])
#     time.sleep(3)
#     d = ego.get_gps().read()
#     time.sleep(1)
#     print(WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)) 
#     p = ego.get_gps().get_location()
#     print(MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2)))
    
    