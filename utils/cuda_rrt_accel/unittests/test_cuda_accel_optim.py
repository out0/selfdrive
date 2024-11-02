import math
import numpy as np
import unittest
import sys
sys.path.append("../../../")
from utils.cudac.cuda_frame import CudaFrame
from utils.cuda_rrt_accel.cuda_rrt_accel import CudaGraph
from model.waypoint import Waypoint

# Equivalent of the compute_dist function
def compute_dist(x1, z1, x2, z2):
    dx = x2 - x1
    dz = z2 - z1
    return math.sqrt(dx * dx + dz * dz)

# Equivalent of the buildWithValue function
def build_with_value(width, height, value):
    img_array = np.full((height, width, 3), value, dtype=np.float32)
    return img_array


# Testing
class TestCudaRRTAccel(unittest.TestCase):
    def test_optimization(self):
        g = CudaGraph(1000, 1000, 0, 0, Waypoint(0, 0), Waypoint(0, 0))
        g.add_point(500, 500, -1, -1, 0)
        cost = 0 + compute_dist(500, 450, 500, 500)
        g.add_point(500, 450, 500, 500, cost)
        cost += compute_dist(400, 350, 500, 450)
        g.add_point(400, 350, 500, 450, cost)
        cost += compute_dist(300, 250, 400, 350)
        g.add_point(300, 250, 400, 350, cost)
        cost += compute_dist(300, 250, 500, 0)
        g.add_point(500, 0, 300, 250, cost)

        self.assertEqual(g.count(), 5)

        parent = g.get_parent(500, 0)
        self.assertEqual(parent, (300, 250))

        # Add a new point with a better parent
        g.add_point(500, 100, 500, 450, compute_dist(500, 100, 500, 450))

        # Optimize the graph with this point
        img_array = build_with_value(1000, 1000, 1)
        cuda_frame = CudaFrame(img_array, 0, 0, Waypoint(0, 0), Waypoint(0, 0))
        cuda_frame.set_goal_vectorized(Waypoint(1000, 1))

        g.optimize_graph(cuda_frame, 500, 100, 500, 450, compute_dist(500, 100, 500, 450), 120)

        parent = g.get_parent(500, 0)
        self.assertEqual(parent, (500, 100))

if __name__ == "__main__":
    unittest.main()
