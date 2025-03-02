import sys, time
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest, math
from fast_rrt.fastrrt import FastRRT
import numpy as np
from cudac.cuda_frame import CudaFrame, Waypoint
import cv2

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000

class TestFastRRT(unittest.TestCase):
    
    def test_search(self):
        return
        rrt = FastRRT(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=119,
            lower_bound_z=148,
            upper_bound_x=137,
            upper_bound_z=108,
            max_path_size_px=20.0,
            dist_to_goal_tolerance_px=20.0,
            libdir=None
        )
        
        raw = np.array(cv2.imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/tests/bev_1.png", cv2.IMREAD_COLOR), dtype=np.float32)

        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=22,
            min_dist_z=40
        )

        self.assertFalse(rrt.is_planning())
        
        path = rrt.get_planned_path()
        self.assertTrue(path is None)

        ptr = frame.get_cuda_frame()

        rrt.set_plan_data(ptr, 128, 0, 0, 1)
        rrt.run()
        
        self.assertTrue(rrt.goal_reached())
        
        path = rrt.get_planned_path()
        
        print(path.shape)


if __name__ == "__main__":
    unittest.main()