import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
from pydriveless import SearchFrame, float3
from pydriveless import Waypoint, angle
import cv2
from test_utils import fix_cv2_import
fix_cv2_import()

CPU_THRESHOLD = 20

class TestSearchFrameCheckFeasiblePath(unittest.TestCase):
        
    def __add_z_line(raw_frame: np.array, x: int):
        for z in range(raw_frame.shape[0]):
            raw_frame[z, x] = [1.0, 0.0, 0.0]


    def test_feasible_path(self):
        f1 = SearchFrame(100, 100, (-1, -1), (-1, -1))
        costs = np.array([0.0, -1.0], np.float32)
        f1.set_class_costs(costs)
        
        raw_frame = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        TestSearchFrameCheckFeasiblePath.__add_z_line(raw_frame, x=30)
        TestSearchFrameCheckFeasiblePath.__add_z_line(raw_frame, x=70)
        f1.set_frame_data(raw_frame)

        x = 34
        path = []
        for z in range(CPU_THRESHOLD - 1):
            path.append(Waypoint(x, z, heading=angle.new_deg(0.0)))
        
        res = f1.check_feasible_path(min_distance=(10, 10), path=path)
        self.assertFalse(res)

        x = 36
        path = []
        for z in range(CPU_THRESHOLD - 1):
            path.append(Waypoint(x, z, heading=angle.new_deg(0.0)))
        
        res = f1.check_feasible_path(min_distance=(10, 10), path=path)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
