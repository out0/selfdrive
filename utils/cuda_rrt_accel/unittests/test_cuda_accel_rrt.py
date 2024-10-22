import sys, time
sys.path.append("../")
import unittest, math, numpy as np
from cuda_rrt_accel import CudaGraph

class TestCudaAccellRRT(unittest.TestCase):
     
    def test_add_count(self):
        
        g = CudaGraph(1000, 1000)
        g.add_point(100, 100, 0, 0, 1000.12)
        g.add_point(100, 101, 0, 0, 1000.12)

        self.assertFalse(g.check_in_graph(900, 900))
        self.assertTrue(g.check_in_graph(100, 100))

        self.assertEqual(g.count(), 2)

    def test_find_best_neighbor(self):
        g = CudaGraph(1000, 1000)
        g.add_point(100, 100, -1, -1, 0)
        g.add_point(100, 70, -1, -1, 0)
        g.add_point(130, 50, -1, -1, 0)

        res = g.find_best_neighbor(110, 100, 1.0)

        self.assertEqual(res, None)

        res = g.find_best_neighbor(110, 100, 10.0)
        self.assertEqual(res[0], 100)
        self.assertEqual(res[1], 100)

        res = g.find_best_neighbor(120, 80, 1000.0)
        self.assertEqual(res[0], 100)
        self.assertEqual(res[1], 70)


if __name__ == "__main__":
    unittest.main()
