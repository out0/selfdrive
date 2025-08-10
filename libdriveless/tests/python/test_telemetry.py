import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import SearchFrame, float3
from pydriveless import Telemetry
import os
import cv2
from test_utils import fix_cv2_import
fix_cv2_import()

class TestTelemetry(unittest.TestCase):
        
    def test_produce_autoconsume(self):
        
        if os.path.exists("test1.png"): os.remove("test1.png")
        if os.path.exists("test2.png"): os.remove("test2.png")
        if os.path.exists("test3.txt"): os.remove("test3.txt")

        msg = "Hello World"
        #Telemetry.terminate()
        Telemetry.log("test3.txt", msg)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [1, 0, 0]

        og = SearchFrame(width=100, height=100, lower_bound=(-1, -1), upper_bound=(-1, -1))
        og.set_class_colors(np.array([
            [0, 0, 0],
            [255, 0, 0]
        ]))
        og.set_frame_data(img)       
        Telemetry.log("test1.png", og.get_color_frame())
        Telemetry.log("test2.png", og)
        
        while not Telemetry.empty():
            time.sleep(0.1)

        time.sleep(0.5)

        with open("test3.txt") as f:
            v = f.readlines()
            self.assertEqual(v[0], f"{msg}\n")

        d1 = np.array(cv2.imread("test1.png"))
        for z in range(100):
            for x in range(100):
                if (z >= 25 and z < 75) and (x >= 25 and x < 75):
                    if d1[x, z, 0] != 255:
                        self.fail(f"wrong value at ({x}, {z}): {d1[x, z, 0]}")
                else:
                    if d1[x, z, 0] != 0:
                        self.fail(f"wrong value at ({x}, {z}): {d1[x, z, 0]}")


        d2 = np.array(cv2.imread("test2.png"), dtype=np.float32)
        for z in range(100):
            for x in range(100):
                if (z >= 25 and z < 75) and (x >= 25 and x < 75):
                    if d2[x, z, 0] != 1.0:
                        self.fail(f"wrong value at ({x}, {z}): {d2[x, z, 0]}")
                else:
                    if d2[x, z, 0] != 0.0:
                        self.fail(f"wrong value at ({x}, {z}): {d2[x, z, 0]}")

        Telemetry.terminate()



if __name__ == "__main__":
    unittest.main()
