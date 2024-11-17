import sys, time
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
from model.map_pose import MapPose
from planner.physical_model import ModelCurveGenerator
from model.waypoint import Waypoint
import os
if not os.path.exists("/usr/local/lib/libfast-rrts.so"):
   exit(0)

from fast_rrt import FastRRT
from testing.test_utils import TestFrame





class TestFastRRTCurveGen(unittest.TestCase):
    
    def test_compare_curve_with_vel_angle_size(self):
        pygen = ModelCurveGenerator()
        
        start = Waypoint(128, 128, 0)
        v = 2.0
        a = 30
        path_size = 100
        
        points = pygen.gen_path_waypoint(
            start,
            v, 
            a,
            path_size
        )
        
        rrt = FastRRT()
        points2 = rrt.gen_path_waypoint(
            start,
            v,
            a,
            path_size
        )
               
        for i in range(len(points)):
            self.assertTrue(abs(points[i].x - points2[i].x <= 2))
            self.assertTrue(abs(points[i].z - points2[i].z <= 2))
            self.assertTrue(abs(points[i].heading - points2[i].heading) <= 1)
        
        
    def test_compare_curve_start_end(self):
        pygen = ModelCurveGenerator()
        
        start = Waypoint(128, 128, 0)
        end = Waypoint(230, 14)
        v = 2.0
        a = 30
        path_size = 100
        
        points = pygen.connect_nodes_with_path(
            start,
            end, 
            v,
        )
        
        # frame = TestFrame(256, 256, 255)
        # frame.add_path(points, color=[0, 0, 255])
        # frame.dump_to_file("debug_curve_cpp.png")
        
        rrt = FastRRT()
        points2 = rrt.connect_nodes_with_path(
            start,
            end,
            v
        )

        # frame.add_path(points2, color=[0, 0, 0])
        # frame.dump_to_file("debug_curve_cpp.png")
        
               
        for i in range(len(points)):
            self.assertTrue(abs(points[i].x - points2[i].x <= 2))
            self.assertTrue(abs(points[i].z - points2[i].z <= 2))
            self.assertTrue(abs(points[i].heading - points2[i].heading) <= 1)
        
        

if __name__ == "__main__":
    if os.path.exists("/usr/local/lib/libfast-rrts.so"):
       unittest.main()


