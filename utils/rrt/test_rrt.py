import sys, time
sys.path.append("../../")
import unittest, math
import numpy as np
import cv2
from rrt import RRT
from test_utils import TestFrame, TestData, TestUtils


OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000

class TestRRT(unittest.TestCase):
    
    def dev_frame(self) -> np.ndarray:
        return np.full((256, 256, 3), 1.0, dtype=np.float32)
    
    def test_regular_rrt(self):
        
        img = TestUtils.timed_exec(self.dev_frame)
        
        
        TestUtils.output_path_result(img, None, "output1.png")
        rrt = RRT(
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
            max_path_size_px=40.0,
            dist_to_goal_tolerance_px=15.0
            )
        
        rrt.set_plan_data(img, (128, 128, 0), (128, 0, 0), 1)
        
        start_time = time.time()
        rrt.search_init()
        while (not rrt.goal_reached() and rrt.loop(False)):
            pass
            
        end_time = time.time()

        #self.assertTrue(rrt.goal_reached())
        print(f"goal reached? {rrt.goal_reached()}")

        execution_time = end_time - start_time  # Calculate the time taken
        print(f"Coarse path: {1000*execution_time:.6f} ms")
        
    
    
if __name__ == "__main__":
    unittest.main()