import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import SearchFrame, float3

import cv2
from test_utils import fix_cv2_import
fix_cv2_import()


class TestSearchFrameCopyback(unittest.TestCase):
        
    def test_copy_back(self):
        frame = SearchFrame(100, 100, (-1, -1), (-1, -1))
        data = np.full((100, 100, 3), 1.0, np.float32)
        frame.set_frame_data(data)
        frame.set_class_costs(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        frame[(30, 30)] = float3(2.0, 1.0, 1.0)
        
        f = frame.get_frame()
        for z in range(100):
            for x in range(100):
                if x == 30 and z == 30:
                    self.assertAlmostEqual(2.0, f[z, x, 0])
                    self.assertAlmostEqual(1.0, f[z, x, 1])
                    self.assertAlmostEqual(1.0, f[z, x, 2])
                elif f[z, x, 0] != 1.0 or\
                    f[z, x, 1] != 1.0 or\
                    f[z, x, 2] != 1.0:
                        self.fail(f"pos {x}, {x} is wrong: should be (1.0, 1.0, 1.0) but it is {f[z, x]}")
        
        frame.process_distance_to_goal(100, 0)
        f = frame.get_frame()
        for z in range(100):
            for x in range(100):
                dx = x - 100
                dz = z - 0
                dist_to_goal = math.sqrt(dx * dx + dz * dz)
                if x == 30 and z == 30:
                    self.assertAlmostEqual(2.0, f[z, x, 0])
                    self.assertAlmostEqual(dist_to_goal, f[z, x, 1], places=3)
                    self.assertAlmostEqual(1.0, f[z, x, 2])
                elif f[z, x, 0] != 1.0 or\
                    f[z, x, 1] - dist_to_goal > 0.01 or\
                    f[z, x, 2] != 1.0:
                        self.fail(f"pos {x}, {x} is wrong: should be (1.0, {dist_to_goal}, 1.0) but it is {f[z, x]}")       


if __name__ == "__main__":
    unittest.main()
