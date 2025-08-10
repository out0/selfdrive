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

def catmull_roll_interpolate(p0: Waypoint, p1: Waypoint, p2: Waypoint, p3: Waypoint, resolution: int) -> list[tuple[float, float, float]]:    
        res = []
        for t in np.linspace(0, 1, resolution):
            #print (f"t = {t}")
            t2, t3 = t*t, t*t*t
            x = 0.5 * ((2*p1.x) + (-p0.x + p2.x) * t +
                    (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * t2 +
                    (-p0.x + 3*p1.x - 3*p2.x + p3.x) * t3)
            y = 0.5 * ((2*p1.z) + (-p0.z + p2.z) * t +
                    (2*p0.z - 5*p1.z + 4*p2.z - p3.z) * t2 +
                    (-p0.z + 3*p1.z - 3*p2.z + p3.z) * t3)
            
            res.append((x, y, 0))
        return res


def catmull_roll_spline_interpolation(path: list[Waypoint], resolution: int = 10) -> list[tuple[float, float, float]]:    
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


from scipy.signal import savgol_filter

class TestPathHeading(unittest.TestCase):


    def compute_headings_smoothed(points, window_length=7, polyorder=3):
        """
        Computes heading angles from x, y points using centered finite differences,
        after smoothing with Savitzky-Golay filter.

        Parameters:
        - points: list of (x, y) tuples
        - window_length: length of the smoothing filter window (must be odd)
        - polyorder: order of the polynomial for smoothing

        Returns:
        - list of (x, y, heading_deg)
        """
        x = points[:, 0]
        y = points[:, 1]

        # Smooth x and y
        if len(points) < window_length:
            window_length = max(3, len(points) // 2 * 2 + 1)  # make odd and safe
        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)

        # Compute centered derivatives
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)

        # Compute headings
        heading_rad = HALF_PI - np.arctan2(-dy, dx)
        heading_deg = np.degrees(heading_rad)

        return list(zip(x, y, heading_deg))






    def test_path_heading_against_hermite(self):
        p1 = Waypoint(x=50, z=99, heading=angle.new_deg(0))
        p2 = Waypoint(x=0, z=0, heading=angle.new_deg(-45))

        points = Interpolator.hermite(100, 100, p1, p2, return_as_waypoint=False)
        resolution = 10
        frame = np.full((100, 100, 3), 0, dtype=np.uint8)

        #cubic_points = Interpolator.cubic_spline(points, resolution=resolution)

        # cubic_points = catmull_roll_spline_interpolation(points, resolution)
        # for i in range(len(cubic_points) - 1):
        #     dx = cubic_points[i+1][0] - cubic_points[i][0]
        #     dz = cubic_points[i+1][1] - cubic_points[i][1]
        #     cubic_points[i] = (cubic_points[i][0], cubic_points[i][1],  math.degrees(HALF_PI - math.atan2(-dz, dx)))
        

        lst = TestPathHeading.compute_headings_smoothed(points)
        i = 0
        for p in lst:
            print (f"{int(p[0])}, {int(p[1])}, {float(p[2]):.2f} x {math.degrees(float(points[i, 2])):.2f}")
            i +=1
        # for p in points:
        #     frame[p.z, p.x] = [255, 255, 255]
        
        # size = len(cubic_points)
        # i = 0
        # j = 0
        # while i < size:
        #     p = cubic_points[i]
        #     # if p.z < 0 or p.z >= frame.shape[0]: continue
        #     # if p.x < 0 or p.x >= frame.shape[1]: continue
        #     # frame[p.z, p.x] = [0, 255, 0]

        #     ph = points[j]
        #     print(f"({ph.x}, {ph.z}, {ph.heading.deg():.2f}) heading_approx = {lst[j]}")
        #     i += resolution
        #     j += 1
        

        # cv2.imwrite("debug.png", frame)

            
        


if __name__ == "__main__":
    unittest.main()
        


