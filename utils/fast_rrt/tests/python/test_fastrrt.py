import sys, time
sys.path.append("../../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/driveless-new/libfastrrt")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/driveless-new/libdriveless")
sys.path.append("../")
import unittest, math
from pydriveless.angle import Angle
from pyfastrrt.fastrrt import FastRRT


class TestFastRRT(unittest.TestCase):
    
    def test_init(self):
        rrt = FastRRT(256, 256, 32.4, 32.4, Angle.new_rad(0.45), 2.3, 1000, 30, 5, libdir="/home/cristiano/Documents/Projects/Mestrado/code/driveless-new/libfastrrt/libfastrrt.so")
        pass
    


if __name__ == "__main__":
    unittest.main()