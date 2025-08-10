import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import Interpolator
from pydriveless import Waypoint
from pydriveless import angle



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

    def test_interpolator_straight(self):
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

    def test_interpolator_straight2(self):
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

            
if __name__ == "__main__":
    unittest.main()
        


