import unittest
from pyfastrrt import FastRRT
from pydriveless import SearchFrame
import time, math
import cv2, numpy as np

MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 60000
TIMEOUT = -1

PERCEPTION_WIDTH_M = 1
PERCEPTION_HEIGHT_M = 1

class TestFastRRTFrenetix(unittest.TestCase):
    
    def test_execute_scenario(self):
        
        img = np.array(cv2.imread("converted_bev_23.png"), dtype=np.float32)

       
        
        frame = SearchFrame (
            width=img.shape[1],
            height=img.shape[0],
            lower_bound=(-1, -1),
            upper_bound=(-1, -1),
        )
    
        frame.set_class_costs(np.array([-1, 0, 0, 0, 0]))
        frame.set_class_colors((np.array([
            [0, 0, 0],
            [128, 128, 128],
            [0, 0, 255],
            [255, 255, 255],
            [255, 0, 0]
        ])))
        frame.set_frame_data(img)

        
        rrt = FastRRT(
            search_frame=frame,
            perception_height_m=PERCEPTION_HEIGHT_M,
            perception_width_m=PERCEPTION_WIDTH_M,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=2,
            min_dist_z=2,
            max_path_size_px=30,
            dist_to_goal_tolerance_px=20,
            path_costs=np.array([-1, 0, 0, 0, 0])
        )
        
        start = (416, 686, -0.039754376)
        goal = (296, 15, 1.9513413283239596)
        
        rrt.set_plan_data(
            frame,
            start=start,
            goal=goal,
            velocity_m_s=1.0
        )
        
        
        start_time = time.time()

        frame.process_safe_distance_zone(min_distance=(1,1), compute_vectorized=False)
        frame.process_distance_to_goal(296, 15)


        rrt.search_init(True)
        loop_count = 0
        
        # cuda_fr: CudaFrame = data.frame
        # cuda_fr.invalidate_cpu_frame()
        # fr = data.frame.get_frame()

        while not rrt.goal_reached() and rrt.loop(True):
            loop_count += 1
            nodes = rrt.export_graph_nodes()     
            TestUtils.output_path_result(data.frame, nodes, "output1.png")
        end_time = time.time()
        execution_time = end_time - start_time
        
        path = rrt.get_planned_path(interpolate=True)
        if path is None:
            print(f"no path found")
            return False
        
        print (f"found path with {len(path)} waypoints in {1000*execution_time:.2f} ms")
    

if __name__ == "__main__":
    unittest.main()