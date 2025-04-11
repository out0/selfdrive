import sys
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest
from utils.fast_rrt.fastrrt import FastRRT
from test_utils import TestFrame, TestData, TestUtils
import time


MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = -1

class TestFastRRT(unittest.TestCase):
    def test_fast_rrt_with_custom1(self):
        # Load the test image
        data = TestFrame(file="custom1.png").get_test_data()

        
        rrt = FastRRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=data.lower_bound.x,
            lower_bound_z=data.lower_bound.z,
            upper_bound_x=data.upper_bound.x,
            upper_bound_z=data.upper_bound.z,
            max_path_size_px=50.0,
            dist_to_goal_tolerance_px=15.0,
            libdir=None
        )


        rrt.set_plan_data(
            data.frame.get_cuda_frame(),
            start=(data.start.x, data.start.z, data.start.heading),
            goal=(data.goal.x, data.goal.z, data.goal.heading),
            velocity_m_s=1.0
        )
        
        rrt.search_init()
        
        start_time = time.time()
        while not rrt.goal_reached() and rrt.loop():
            #TestUtils.log_graph(rrt, data.frame, "output1.png")
            pass
        end_time = time.time()

        execution_time = end_time - start_time  # Calculate the time taken
        print(f"Coarse path: {1000*execution_time:.6f} ms")
            
            
        # Check if the goal is reached
        self.assertTrue(rrt.goal_reached())
        
        path = rrt.get_planned_path(interpolate=True)
               
        print ("path size: ", len(path))
        TestUtils.output_path_result(data.frame, path, "output1.png")
        
        # print ("optimizing")
        # for _ in range(30):
        #     rrt.loop_optimize()
        #     path = rrt.get_planned_path(interpolate=True)
        #     TestUtils.output_path_result(data.frame, path, "output1.png")
            
        TestUtils.output_path_result(data.frame, path, "output1.png")

if __name__ == "__main__":
    unittest.main()