import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import angle
from pydriveless import MapPose

import cv2
from test_utils import fix_cv2_import
fix_cv2_import()



class TestMapPose(unittest.TestCase):

    def test_map_pose_encode_decode(self):
        pose1 = MapPose(10, -10.2, 23.3456, 10.2)
        p = str(pose1)
        pose2 = MapPose.from_str(p)
        
        self.assertEqual(pose1.x, pose2.x)
        self.assertEqual(pose1.y, pose2.y)
        self.assertEqual(pose1.z, pose2.z)
        self.assertTrue(pose1.heading == pose2.heading)
        

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

    def test_distance_to_line(self):
        line_p1 = MapPose(2, 2, 0)
        line_p2 = MapPose(6, 6, 0)
        p = MapPose(4, 0, 0)
        
        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 2*math.sqrt(2), places=3)
        
        p = MapPose(6, 0, 0)
        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 3*math.sqrt(2), places=3)

    def test_compute_path_heading(self):
        line_p1 = MapPose(2, 2, 0)
        line_p2 = MapPose(6, 6, 0)
                
        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertAlmostEqual(heading.rad(), math.pi/4, places=3)
        
        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertAlmostEqual(heading.rad(), -math.pi/4 - math.pi/2, places=3)
        
        line_p1 = MapPose(0, 0, 0)
        line_p2 = MapPose(0, 6, 0)
        
        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertAlmostEqual(heading.rad(), math.pi/2, places=3)
        
        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertAlmostEqual(heading.rad(), -math.pi/2, places=3)
    
    def test_equality_operation(self):
        p1 = MapPose(2, 2, 0)
        p2 = MapPose(2, 2, 0)
        
        self.assertTrue(p1 == p2)
        
        p2.heading.set_deg(1)
        self.assertFalse(p1 == p2)
        
        p2 = MapPose(1, 2, 0)
        self.assertFalse(p1 == p2)
        
        p2 = MapPose(2, 3, 0)
        self.assertFalse(p1 == p2)

        p2 = MapPose(2, 2, 1)
        self.assertFalse(p1 == p2)
      
    
    def test_project_on_path(self):
        line_p1 = MapPose(2, 2, 0)
        line_p2 = MapPose(6, 6, 0)
        p = MapPose(4, 0, 0)
        
        projected, dist, path_size = MapPose.project_on_path(line_p1, line_p2, p)
        
        self.assertAlmostEqual(dist, 0, places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(2, 2, 0), projected)
        
        p = MapPose(6, 0, 0)
        
        projected, dist, path_size = MapPose.project_on_path(line_p1, line_p2, p)
        
        self.assertAlmostEqual(dist,  math.sqrt(2), places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(3, 3, 0), projected)


    def test_project_on_path(self):
        line_p1 = MapPose(2, 2, 0, angle.new_rad(0))
        line_p2 = MapPose(6, 6, 0, angle.new_rad(0))
        p = MapPose(4, 0, 0, angle.new_rad(0))

        pose, path_size, distance_from_p1 = MapPose.project_on_path(line_p1, line_p2, p)

        self.assertAlmostEqual(distance_from_p1, 0.0, places=3)
        self.assertAlmostEqual(path_size, math.sqrt(32), places=3)
        self.assertEqual(pose, MapPose(2, 2, 0, angle.new_rad(0)))

        p2 = MapPose(6, 0, 0, angle.new_rad(0))
        pose, path_size, distance_from_p1 = MapPose.project_on_path(line_p1, line_p2, p2)

        self.assertAlmostEqual(distance_from_p1, math.sqrt(2), places=3)
        self.assertAlmostEqual(path_size, math.sqrt(32), places=3)
        self.assertEqual(pose, MapPose(3, 3, 0, angle.new_rad(0)))

    def test_remove_repeated_seq_elements_x(self):
        lst = []
        for x in range(10):
            lst.append(MapPose(x, 1, 1, angle.new_deg(0)))
            lst.append(MapPose(x, 1, 1, angle.new_deg(0)))

        self.assertEqual(len(lst), 20)
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].x, float(i))

        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].x, float(i))

        lst.clear()
        for x in range(10):
            lst.append(MapPose(x, 1, 1, angle.new_deg(0)))
        for x in range(10):
            lst.append(MapPose(x, 1, 1, angle.new_deg(0)))
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 20)

    def test_remove_repeated_seq_elements_y(self):
        lst = []
        for y in range(10):
            lst.append(MapPose(1, y, 1, angle.new_deg(0)))
            lst.append(MapPose(1, y, 1, angle.new_deg(0)))

        self.assertEqual(len(lst), 20)
        lst =  MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].y, float(i))

        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].y, float(i))

        lst.clear()
        for y in range(10):
            lst.append(MapPose(1, y, 1, angle.new_deg(0)))
        for y in range(10):
            lst.append(MapPose(1, y, 1, angle.new_deg(0)))
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 20)

    def test_remove_repeated_seq_elements_z(self):
        lst = []
        for z in range(10):
            lst.append(MapPose(1, 1, z, angle.new_deg(0)))
            lst.append(MapPose(1, 1, z, angle.new_deg(0)))

        self.assertEqual(len(lst), 20)
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].z, float(i))

        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].z, float(i))

        lst.clear()
        for z in range(10):
            lst.append(MapPose(1, 1, z, angle.new_deg(0)))
        for z in range(10):
            lst.append(MapPose(1, 1, z, angle.new_deg(0)))
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 20)

    def test_remove_repeated_seq_elements_heading(self):
        lst = []
        for a in range(10):
            lst.append(MapPose(1, 1, 1, angle.new_deg(a)))
            lst.append(MapPose(1, 1, 1, angle.new_deg(a)))

        self.assertEqual(len(lst), 20)
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].heading, angle.new_deg(i), places=6)

        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 10)

        for i in range(10):
            self.assertAlmostEqual(lst[i].heading, angle.new_deg(i), places=6)

        lst.clear()
        for a in range(10):
            lst.append(MapPose(1, 1, 1, angle.new_deg(a)))
        for a in range(10):
            lst.append(MapPose(1, 1, 1, angle.new_deg(a)))
        lst = MapPose.remove_repeated_seq_points_in_list(lst)
        self.assertEqual(len(lst), 20)




if __name__ == "__main__":
    unittest.main()
