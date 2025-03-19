import sys, time
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest, math
from fast_rrt.graph import CudaGraph
import numpy as np
from cudac.cuda_frame import CudaFrame, Waypoint
import cv2

OG_REAL_WIDTH = 34.641016151377535
OG_REAL_HEIGHT = 34.641016151377535
MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 40000
#TIMEOUT = -1


GRAPH_TYPE_NODE = 1
GRAPH_TYPE_TEMP = 2
GRAPH_TYPE_PROCESSING = 3


def plot_costs (costs: np.ndarray, file: str) -> None:
    normalized = (costs / 10.0) * 255  # Scale 0-10.0 to 0-255
    img = normalized.astype(np.uint8)       # Convert to uint8 for cv2

# Save as grayscale PNG
    cv2.imwrite(file, img)
    pass


class TestCudaGraph(unittest.TestCase):
    
    def test_apf(self):
        graph = CudaGraph(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=119,
            lower_bound_z=148,
            upper_bound_x=137,
            upper_bound_z=108,
            libdir=None
        )
        
        raw = np.full((256, 256, 3), 1.0, dtype=float)
        
        raw[128, 128, 0] = 0.0 # add obstacle in center pos
        
        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(-1, -1),
            upper_bound=Waypoint(-1, -1),
            min_dist_x=0,
            min_dist_z=0
        )
        
        graph.compute_apf(frame, 5)
        
        costs = graph.get_intrinsic_costs()
        
        plot_costs(costs, "test.png")

    def test_apf2(self):
        graph = CudaGraph(
            width=256,
            height=256,
            perception_height_m=OG_REAL_HEIGHT,
            perception_width_m=OG_REAL_WIDTH,
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=119,
            lower_bound_z=148,
            upper_bound_x=137,
            upper_bound_z=108,
            libdir=None
        )
        
        raw = np.array(cv2.imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/fast_rrt/tests/bev_1.png", cv2.IMREAD_COLOR), dtype=np.float32)
        
       
        frame = CudaFrame(
            frame=raw,
            lower_bound=Waypoint(119, 148),
            upper_bound=Waypoint(137, 108),
            min_dist_x=0,
            min_dist_z=0
        )
        
        graph.compute_apf(frame, 5)
        
        costs = graph.get_intrinsic_costs()
        
        plot_costs(costs, "test2.png")

if __name__ == "__main__":
    unittest.main()