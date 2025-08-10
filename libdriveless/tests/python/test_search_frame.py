import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import SearchFrame, float3

import cv2
from test_utils import fix_cv2_import
fix_cv2_import()


COLORS = np.array([
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142], # car
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [110, 190, 160],
            [170, 120, 50],
            [55, 90, 80],
            [45, 60, 150],
            [157, 234, 50],
            [81, 0, 81],
            [150, 100, 100],
            [230, 150, 140],
            [180, 165, 180]
        ], dtype=np.uint32)

class TestSearchFrame(unittest.TestCase):
        
    def test_get_set(self):
        frame = SearchFrame(100, 101, (45, 55), (55, 45))
        self.assertEqual(frame.width(), 100)
        self.assertEqual(frame.height(), 101)
        self.assertEqual(frame.lowerBound()[0], 45)
        self.assertEqual(frame.lowerBound()[1], 55)
        self.assertEqual(frame.upperBound()[0], 55)
        self.assertEqual(frame.upperBound()[1], 45)
        
    
    def test_export_color_frame(self):
        frame = SearchFrame(100, 101,  (45, 55), (55, 45))
        frame.set_class_colors(COLORS)       
        data = np.zeros((100, 100, 3), dtype=np.float32)
        c = 0
        for i in range(100):
            for j in range(100):
                data[i, j] = c
                c = (c + 1) % 29
                
            
        frame.set_frame_data(data)
        f = frame.get_color_frame()
        
        for i in range(100):
            for j in range(100):
                for k in range(3):
                    f[i, j, k] == COLORS[c][k]
                c = (c + 1) % 29

    # def test_export_cuda_frame(self):
    #     frame = SearchFrame(100, 101,  (45, 55), (55, 45))
    #     frame.set_class_colors(COLORS)       
    #     data = np.zeros((100, 100, 3), dtype=np.float32)
    #     c = 0
    #     for i in range(100):
    #         for j in range(100):
    #             data[i, j] = c
    #             c = (c + 1) % 29
                
            
    #     frame.set_frame_data(data)
    #     f = frame.get_frame()
        
    #     for i in range(100):
    #         for j in range(100):
    #             f[i, j, 0] == (c + 1) % 29
                
    
    def test_convert_bev_file(self):
        fin = np.array(cv2.imread("bev_1.png"))
        frame = SearchFrame(fin.shape[1], fin.shape[0], (0, 0), (0, 0))
        frame.set_frame_data(fin)
        frame.set_class_colors(COLORS)
        fout = frame.get_color_frame()
        cv2.imwrite("bev_1_conv.png", fout)
    
    def test_class_costs(self):
        frame = SearchFrame(100, 101, (45, 55), (55, 45))
        costs = np.array([
            1.1, 2.2, 3.3, 4.4, 5.5
        ], np.float32)
        frame.set_class_costs(costs)
        
        self.assertAlmostEqual(frame.get_class_cost(0), 1.1, places=3)
        self.assertAlmostEqual(frame.get_class_cost(1), 2.2, places=3)
        self.assertAlmostEqual(frame.get_class_cost(2), 3.3, places=3)
        self.assertAlmostEqual(frame.get_class_cost(3), 4.4, places=3)
        self.assertAlmostEqual(frame.get_class_cost(4), 5.5, places=3)

    def test_get_set(self):
        frame = SearchFrame(100, 101, (-1, -1), (-1, -1))
        data = np.full((100, 100, 3), 1.0, np.float32)
        frame.set_frame_data(data)
        
        frame[(30, 30)] = float3(2.0, 1.0, 1.0)
        
        for z in range(100):
            for x in range(100):
                if x == 30 and z == 30:
                    self.assertAlmostEqual(2.0, frame[(x, z)].x)
                elif  frame[(x, z)].x != 1.0 or\
                    frame[(x, z)].y != 1.0 or\
                    frame[(x, z)].z != 1.0:
                        self.fail(f"pos {x}, {x} is wrong: should be (1.0, 1.0, 1.0) but it is {frame[(x, z)]}")

    

if __name__ == "__main__":
    unittest.main()
