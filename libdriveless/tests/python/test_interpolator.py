import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import Interpolator, EgoParams, SearchParams
from pydriveless import Waypoint, WorldPose, MapPose
from pydriveless import angle
import cv2


class TestInterpolator(unittest.TestCase):

    def test_hermite_straight(self):
        p1 = Waypoint(x=50, z=99, heading=angle.new_deg(0))
        p2 = Waypoint(x=50, z=0, heading=angle.new_deg(0))

        points = Interpolator.hermite(100, 100, p1, p2)
        self.assertEqual(99, len(points))

        last_z = -1
        for p in points:
            if p.x != 50:
                self.fail(f"{p} should have x=50")
            if p.z == last_z:
                self.fail(f"{p} should not have the same z as the last point")
            if p.heading != angle.new_deg(0):
                self.fail(f"{p} should have heading 0 degrees")
            last_z = p.z 
        
    def test_hermite_reach(self):
        p1 = Waypoint(x=50, z=50, heading=angle.new_deg(0))

        for x in range(100):
            p2 = Waypoint(x=x, z=0, heading=angle.new_deg(0))
            points = Interpolator.hermite(100, 100, p1, p2)
            self.assertTrue(len(points) >= 50)
            self.assertEqual(points[-1].x, x)
            self.assertEqual(points[-1].z, 0)

    def test_interpolator_cubic_spline_straight(self):
        p1 = Waypoint(50, 99, angle.new_deg(0))
        p2 = Waypoint(50, 70, angle.new_deg(0))
        p3 = Waypoint(50, 30, angle.new_deg(0))
        p4 = Waypoint(50, 0, angle.new_deg(0))

        points = [p1, p2, p3, p4]

        res = Interpolator.cubic_spline(points)

        last_z = -1
        for p in res:
            if p.x != 50:
                self.fail(f"{p} should have x=50")
            if p.z == last_z:
                #self.fail(f"{p} should not have the same z as the last point")
                # TODO: melhorar
                print(f"{p} should not have the same z as the last point")
            if p.heading != angle.new_deg(0):
                self.fail(f"{p} should have heading 0 degrees")
            last_z = p.z 

    def test_interpolator_cubic_spline_straight2(self):
        p1 = Waypoint(50, 99, angle.new_deg(10))
        p2 = Waypoint(50, 70, angle.new_deg(10))
        p3 = Waypoint(50, 60, angle.new_deg(10))
        p4 = Waypoint(50, 50, angle.new_deg(10))
        p5 = Waypoint(50, 30, angle.new_deg(10))
        p6 = Waypoint(50, 0, angle.new_deg(10))

        points = [p1, p2, p4, p5, p6]

        res = Interpolator.cubic_spline(points)
        for i in range(0, len(points)-1):
            print(f"{11*i}: {res[11*i]}")
        print(f"{res[43]}")


        last_z = -1
        for p in res:
            if p.x != 50:
                self.fail(f"{p} should have x=50")
            if p.z == last_z:
                # TODO: melhorar
                print(f"{p} should not have the same z as the last point")
            if p.heading != angle.new_deg(0):
                self.fail(f"{p} should have heading 0 degrees")
            last_z = p.z 

    def test_bicycle_model_straight_line(self):
        p1 = Waypoint(50, 99, angle.new_deg(0))
        p2 = Waypoint(50, 0, angle.new_deg(0))

        ego_params = EgoParams.init(100, 100)\
            .with_search_physical_size(10.0, 10.0)\
            .with_vehicle_length(4.5)\
            .with_max_steering_angle(angle.new_deg(40))\
            .with_world_origin(WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0)))\
            .build()
                                               

        search_params = ego_params.new_search_params(start=p1, goal=p2)\
            .with_max_path_size(100)\
            .with_ego_pose(MapPose(x=0, y=0, z=0, heading=angle.new_deg(0)))\
            .build()
        
        res_map, res_og = Interpolator.bicycle_model(ego_params, search_params, steering_angle=angle.new_deg(0), path_size_px=100)

        last_z = -1
        for p in res_og:
            if p.x != 50:
                self.fail(f"{p} should have x=50")
            if p.z == last_z:
                #self.fail(f"{p} should not have the same z as the last point")
                # TODO: melhorar
                print(f"{p} should not have the same z as the last point")
            if p.heading != angle.new_deg(0):
                self.fail(f"{p} should have heading 0 degrees")
            last_z = p.z
      
    def test_bicycle_model_curve_right(self):
        p1 = Waypoint(50, 99, angle.new_deg(0))
        p2 = Waypoint(99, 0, angle.new_deg(0))

        ego_params = EgoParams.init(100, 100)\
            .with_search_physical_size(10.0, 10.0)\
            .with_vehicle_length(4.5)\
            .with_max_steering_angle(angle.new_deg(40))\
            .with_world_origin(WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0)))\
            .build()
                                               

        search_params = ego_params.new_search_params(start=p1, goal=p2)\
            .with_max_path_size(100)\
            .with_ego_pose(MapPose(x=0, y=0, z=0, heading=angle.new_deg(0)))\
            .build()
        
        res_map, res_og = Interpolator.bicycle_model(ego_params, search_params, steering_angle=angle.new_deg(33), path_size_px=100)
       
        reached_end = False
        for p in res_og:
            x = int(p.x)
            z = int(p.z)
            if x == 99 and z == 0:
                reached_end = True

        self.assertTrue(reached_end, "The path should reach the goal point (99, 0)")

       
    def test_bicycle_model_curve_left(self):
        p1 = Waypoint(50, 99, angle.new_deg(0))
        p2 = Waypoint(99, 0, angle.new_deg(0))

        ego_params = EgoParams.init(100, 100)\
            .with_search_physical_size(10.0, 10.0)\
            .with_vehicle_length(4.5)\
            .with_max_steering_angle(angle.new_deg(40))\
            .with_world_origin(WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0)))\
            .build()
                                               

        search_params = ego_params.new_search_params(start=p1, goal=p2)\
            .with_max_path_size(100)\
            .with_ego_pose(MapPose(x=0, y=0, z=0, heading=angle.new_deg(0)))\
            .build()
        
        res_map, res_og = Interpolator.bicycle_model(ego_params, search_params, steering_angle=angle.new_deg(-33), path_size_px=100)
       
        reached_end = False
        for p in res_og:
            x = int(p.x)
            z = int(p.z)
            if x == 0 and z == 0:
                reached_end = True

        self.assertTrue(reached_end, "The path should reach the goal point (99, 0)")

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for p in res_og:
            x = int(p.x)
            y = int(p.z)
            img[y, x] = [0, 255, 0]
        cv2.imwrite("bicycle_model_straight_line.png", img)

if __name__ == "__main__":
    unittest.main()
        


