import unittest
import numpy as np, os, sys
from pydriveless import SearchFrame

def set_class_value(ptr, width, x, z, value):
    ptr[z, x, 0] = value

class TestSearchFrameProcessSafeDistanceZone(unittest.TestCase):
    def test_process_safe_distance_zone_no_obstacles(self):
        f1 = SearchFrame(100, 100, (5, 5), (15, 15))
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if not f1.is_traversable(x, z):
                    self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_single_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_two_pixel_z_line_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 50, 51, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 58:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_three_pixel_z_line_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 50, 51, 1)
        set_class_value(ptr, 100, 50, 52, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 59:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_two_pixel_x_line_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 49, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= 42 and x <= 57 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_three_pixel_x_line_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 49, 50, 1)
        set_class_value(ptr, 100, 51, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= 42 and x <= 58 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_fat_obstacle(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (43-48, 43-48)
        obst_init = 43
        obst_end = 48
        for z in range(obst_init, obst_end + 1):
            for x in range(obst_init, obst_end + 1):
                set_class_value(ptr, 100, x, z, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], False)
        
        for z in range(100):
            for x in range(100):
                if x >= (obst_init-7) and x <= (obst_end+7) and z >= (obst_init-7) and z <= (obst_end+7):
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_single_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_two_pixel_z_line_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 50, 51, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 58:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_three_pixel_z_line_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 50, 51, 1)
        set_class_value(ptr, 100, 50, 52, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= 43 and x <= 57 and z >= 43 and z <= 59:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_two_pixel_x_line_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 49, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= 42 and x <= 57 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_three_pixel_x_line_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (50, 50)
        set_class_value(ptr, 100, 50, 50, 1)
        set_class_value(ptr, 100, 49, 50, 1)
        set_class_value(ptr, 100, 51, 50, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= 42 and x <= 58 and z >= 43 and z <= 57:
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

    def test_process_safe_distance_zone_fat_obstacle_with_vectorize_flag(self):
        f1 = SearchFrame(100, 100, [-1, -1], [-1, -1])
        
        costs = np.array([0.0, -1.0], dtype=np.float32)
        f1.set_class_costs(costs)
        
        SIZE = (100, 100, 3)
        ptr = np.zeros(SIZE, dtype=np.float32)
        
        # adding obstacle to (43-48, 43-48)
        obst_init = 43
        obst_end = 48
        for z in range(obst_init, obst_end + 1):
            for x in range(obst_init, obst_end + 1):
                set_class_value(ptr, 100, x, z, 1)
        f1.set_frame_data(ptr)
        
        f1.process_safe_distance_zone([10, 10], True)
        
        for z in range(100):
            for x in range(100):
                if x >= (obst_init-7) and x <= (obst_end+7) and z >= (obst_init-7) and z <= (obst_end+7):
                    if f1.is_traversable(x, z):
                        self.fail(f"it should NOT be traversable at ({x}, {z})")
                else:
                    if not f1.is_traversable(x, z):
                        self.fail(f"it should be traversable at ({x}, {z})")

if __name__ == '__main__':
    unittest.main()
