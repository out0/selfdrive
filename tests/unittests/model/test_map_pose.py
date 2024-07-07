import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math

from model.map_pose import MapPose

class TestMapPose(unittest.TestCase):

    def test_map_pose_encode_decode(self):
        pose1 = MapPose(10, -10.2, 23.3456, 10.2)
        p = str(pose1)
        pose2 = MapPose.from_str(p)
        
        self.assertEqual(pose1.x, pose2.x)
        self.assertEqual(pose1.y, pose2.y)
        self.assertEqual(pose1.z, pose2.z)
        self.assertEqual(pose1.heading, pose2.heading)
        

    def test_map_pose_distance_between(self):
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,0,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 10, 4)
        
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 10 * math.sqrt(2), 4)
        
        p1 = MapPose(-10,-10,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 20 * math.sqrt(2), 4)
        
        p1 = MapPose(10,10,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 0, 4)

    def test_map_pose_dot(self):
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,10,0,0)
        self.assertAlmostEqual(MapPose.dot(p1, p2), 0, 4)

        p1 = MapPose(10,10,0,0)
        p2 = MapPose(10,10,0,0)
        self.assertAlmostEqual(MapPose.dot(p1, p2), 200, 4)



if __name__ == "__main__":
    unittest.main()
