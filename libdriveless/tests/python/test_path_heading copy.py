import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import Interpolator
from pydriveless import Waypoint
from pydriveless import angle
import cv2
from test_utils import fix_cv2_import
fix_cv2_import()

HALF_PI = math.pi / 2

def catmull_roll_interpolate(p0: Waypoint, p1: Waypoint, p2: Waypoint, p3: Waypoint, resolution: int) -> list[Waypoint]:    
        res = []
        for t in np.linspace(0, 1, resolution):
            print (f"t = {t}")
            t2, t3 = t*t, t*t*t
            x = 0.5 * ((2*p1.x) + (-p0.x + p2.x) * t +
                    (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * t2 +
                    (-p0.x + 3*p1.x - 3*p2.x + p3.x) * t3)
            y = 0.5 * ((2*p1.z) + (-p0.z + p2.z) * t +
                    (2*p0.z - 5*p1.z + 4*p2.z - p3.z) * t2 +
                    (-p0.z + 3*p1.z - 3*p2.z + p3.z) * t3)
            
            dx = 0.5 * ((-p0.x + p2.x) +
                        2 * (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * t +
                        3 * (-p0.x + 3*p1.x - 3*p2.x + p3.x) * t2)
            dy = 0.5 * ((-p0.z + p2.z) +
                        2 * (2*p0.z - 5*p1.z + 4*p2.z - p3.z) * t +
                        3 * (-p0.z + 3*p1.z - 3*p2.z + p3.z) * t2)
            res.append(Waypoint(x, y, angle.new_rad(HALF_PI - np.arctan2(-dy, dx))))
        return res


def catmull_roll_spline_interpolation(path: list[Waypoint], resolution: int = 10) -> list[Waypoint]:
    padded = [path[0]] + path + [path[-1]]
    res = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i+4]
        new_points = catmull_roll_interpolate(p0, p1, p2, p3, resolution)
        res.extend(new_points)
    return res

def catmull_roll_spline_interpolation2(path: list[Waypoint], resolution: int = 10) -> list[Waypoint]:
    res = []
    size = len(path)
    
    new_points = catmull_roll_interpolate(path[0], path[0], path[1], path[2], resolution)
    res.extend(new_points)

    if size >= 4:
        for i in range(len(path) - 3):
            p0, p1, p2, p3 = path[i:i+4]
            new_points = catmull_roll_interpolate(p0, p1, p2, p3, resolution)
            res.extend(new_points)

    new_points = catmull_roll_interpolate(path[size-3], path[size-2], path[size-1], path[size-1], resolution)
    res.extend(new_points)


    return res



class TestPathHeading(unittest.TestCase):


    def test_isolated_interpolation(self):
        p1 = Waypoint(50, 99, angle.new_deg(10))
        p2 = Waypoint(50, 70, angle.new_deg(20))
        p3 = Waypoint(50, 30, angle.new_deg(30))
        p4 = Waypoint(50, 0, angle.new_deg(40))

        path = []
        path.append(p1)
        path.append(p2)
        path.append(p3)
        path.append(p4)

        res = Interpolator.cubic_spline(path, resolution=10)
        #res = catmull_roll_spline_interpolation(path, resolution=10)

        for p in res:
            print (f"{p.x}, {p.z}, {p.heading.deg()}")

         

    def test_path_heading_against_hermite(self):
        p1 = Waypoint(x=50, z=99, heading=angle.new_deg(0))
        p2 = Waypoint(x=0, z=0, heading=angle.new_deg(-45))

        points = Interpolator.hermite(100, 100, p1, p2)
        resolution = 10
        frame = np.full((100, 100, 3), 0, dtype=np.uint8)

        #cubic_points = Interpolator.cubic_spline(points, resolution=resolution)

        cubic_points = catmull_roll_spline_interpolation(points, resolution)
        
        for p in points:
            frame[p.z, p.x] = [255, 255, 255]
        
        size = len(cubic_points)
        i = 0
        j = 0
        while i < size:
            p = cubic_points[i]
            if p.z < 0 or p.z >= frame.shape[0]: continue
            if p.x < 0 or p.x >= frame.shape[1]: continue
            frame[p.z, p.x] = [0, 255, 0]

            ph = points[j]
            print(f"Hermite: ({ph.x}, {ph.z}, {ph.heading.deg():.2f}),  Heading compute: ({p.x}, {p.z}, {p.heading.deg():.2f})")
            i += resolution
            j += 1
        

        cv2.imwrite("debug.png", frame)

            
        


if __name__ == "__main__":
    unittest.main()
        


