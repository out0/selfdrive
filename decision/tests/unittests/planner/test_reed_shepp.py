import sys, time
sys.path.append("../../../")
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import SearchFrame
from pydriveless import CoordinateConverter
from ensemble import ReedsShepp, PhysicalParameters
import cv2
import numpy as np
import unittest

class TestLPOvertaker(unittest.TestCase):

    TIMEOUT_MS = -1
    ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

    def test_bev_1(self):
        
        bev = np.array(cv2.imread("bev_1.png"), dtype=np.float32)
        og = SearchFrame(width=bev.shape[1], height=bev.shape[0], lower_bound=PhysicalParameters.EGO_LOWER_BOUND, upper_bound=PhysicalParameters.EGO_UPPER_BOUND)
        
        og.set_frame_data(bev)
        og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        og.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)

        f = og.get_color_frame()
        cv2.imwrite("debug.png", f)

        start = Waypoint(128, 115, angle.new_deg(20))
        goal =  Waypoint(108, 0, angle.new_deg(0))
        
        #curve = ReedsShepp(step=0.01, max_curv=0.02) #max_curv=0.03356
        curve = ReedsShepp(step=0.01, vehicle_length_m=PhysicalParameters.VEHICLE_LENGTH_M, max_steering_angle=PhysicalParameters.MAX_STEERING_ANGLE, speed=5.0)
        _, _, x_list, y_list, yaw_list = curve.generation(start, goal)

        for j in range(len(x_list)):
            z = int(y_list[j])
            x = int(x_list[j])
            f[z, x, :] = [255, 255, 255]

        cv2.imwrite("debug.png", f)


if __name__ == "__main__":
    unittest.main()
