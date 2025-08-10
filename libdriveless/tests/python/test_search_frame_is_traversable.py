import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
from pydriveless import SearchFrame, float3
from pydriveless import Waypoint, angle

CPU_THRESHOLD = 20

class TestSearchFrameIsTraversable(unittest.TestCase):
        
    def test_all_traversable(self):
        f1 = SearchFrame(100, 100, (-1, -1), (-1, -1))
        costs = np.array([0.0, -1.0], np.float32)
        f1.set_class_costs(costs)
        
        raw_frame = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        f1.set_frame_data(raw_frame)

        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertTrue(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertTrue(res)

        f1.process_safe_distance_zone((10, 10), compute_vectorized=False)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertTrue(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertTrue(res)
        
        f1.process_safe_distance_zone((10, 10), compute_vectorized=True)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertTrue(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertTrue(res)


    def test_not_traversable(self):
        f1 = SearchFrame(100, 100, (-1, -1), (-1, -1))
        costs = np.array([0.0, -1.0], np.float32)
        f1.set_class_costs(costs)
        
        raw_frame = np.full((100, 100, 3), fill_value=1.0, dtype=np.float32)
        f1.set_frame_data(raw_frame)

        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertFalse(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertFalse(res)

        f1.process_safe_distance_zone((10, 10), compute_vectorized=False)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertFalse(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertFalse(res)
        
        f1.process_safe_distance_zone((10, 10), compute_vectorized=True)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=True)
        self.assertFalse(res)
        res = f1.is_traversable(0, 0, heading=angle.new_deg(0), precision_check=False)
        self.assertFalse(res)        

if __name__ == "__main__":
    unittest.main()
