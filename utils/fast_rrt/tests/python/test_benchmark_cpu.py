import sys
sys.path.append("../../")
sys.path.append("../../../")
import unittest
from fast_rrt.cpu_rrt import *
from test_utils import TestFrame, TestData, TestUtils
from fast_rrt.graph import CudaFrame, CudaGraph
import time
import numpy as np


SEGMENTATION_COST = [-1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1]
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = -1

class TestFastRRT(unittest.TestCase):
    
    
    
    def test_small_cluttered(self):
        # Load the test image
        
        #data = TestUtils.timed_exec(TestFrame("small_cluttered_2.png").get_data_cpu)
        f = TestFrame("small_cluttered_1.png")
        data: TestData = TestUtils.timed_exec(f.get_data_cpu)
        data_gpu: TestData = TestUtils.timed_exec(f.get_data_cuda)
        
        proc = TestUtils.pre_process_gpu(data, data_gpu.frame, MAX_STEERING_ANGLE, VEHICLE_LENGTH_M)
        
        rrt = RRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1,
            max_path_size_px=30.0,
            dist_to_goal_tolerance_px=15.0,
            class_cost=SEGMENTATION_COST
        )


        rrt.set_plan_data(proc, data.start, data.goal, 1)

        
        start_time = time.time()
        loop_count = 0
        rrt.search_init(MIN_DIST_GPU)
        while not rrt.goal_reached() and rrt.loop_rrt(True):
            loop_count += 1
            #nodes = rrt.list_nodes()
            #TestUtils.output_path_result_cpu(data.frame, nodes, "output1.png")
            pass
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[Coarse path] total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
        
        
        # Check if the goal is reached
        self.assertTrue(rrt.goal_reached())
        
        path = rrt.get_planned_path(interpolate=True)
        TestUtils.output_path_result_cpu(data.frame, path, "output1.png")
    
    

if __name__ == "__main__":
    unittest.main()