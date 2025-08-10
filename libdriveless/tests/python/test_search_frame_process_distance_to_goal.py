import unittest, math, numpy as np
from pydriveless import SearchFrame

class TestSearchFrameDistanceToGoal(unittest.TestCase):
        
    def test_distance_to_goal(self):
        f1 = SearchFrame(100, 100, (-1, -1), (-1, -1))
        costs = np.array([0.0, -1.0], np.float32)
        f1.set_class_costs(costs)
        
        raw_frame = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        f1.set_frame_data(raw_frame)
   
        f1.process_distance_to_goal(50, -100)

        for z in range(100):
            for x in range(100):
                dist = f1.get_distance_to_goal(x, z)
                dx = x - 50
                dz = z + 100
                d = math.sqrt(dx * dx + dz * dz)
                if abs(d - dist) > 0.01:
                    self.fail()

if __name__ == "__main__":
    unittest.main()
