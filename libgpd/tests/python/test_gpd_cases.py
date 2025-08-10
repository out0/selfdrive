import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import angle
from pydriveless import MapPose, WorldPose, Waypoint
from pydriveless import CoordinateConverter
from pygpd import GoalPointDiscover
from pydriveless import SearchFrame
from test_utils import SEGMENTATION_CLASS_COST, SEGMENTED_COLORS


import cv2
from test_utils import fix_cv2_import
fix_cv2_import()

OG_WIDTH = 256
OG_HEIGHT = 256
OG_REAL_WIDTH: float = 34.641016151377535
OG_REAL_HEIGHT: float = 34.641016151377535
MIN_DISTANCE_WIDTH_PX: int = 22
MIN_DISTANCE_HEIGHT_PX: int = 40
EGO_LOWER_BOUND: tuple[int, int] = (119, 148) 
EGO_UPPER_BOUND: tuple[int, int] =  (137, 108)
    
COORD_ORIGIN = WorldPose(lat=angle.new_deg(-4.303359446566901e-09), 
                      lon=angle.new_deg(-1.5848012769283334e-08),
                      alt=angle.new_deg(1.0149892568588257),
                      compass=angle.new_rad(0))

class TestGoalPointDiscover(unittest.TestCase):
    conv: CoordinateConverter
    discover: GoalPointDiscover

    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.discover = None

    def get_gpd(self) -> GoalPointDiscover:
        if self.discover is None:
            self.conv = CoordinateConverter(
                width=OG_WIDTH,
                height=OG_HEIGHT,
                perceptionWidthSize_m=OG_REAL_WIDTH,
                perceptionHeightSize_m=OG_REAL_HEIGHT,
                origin=COORD_ORIGIN
            )
            self.discover = GoalPointDiscover(self.conv)
        return self.discover

    def basic_test_case(self, file: str) -> SearchFrame:
        f1 = SearchFrame(OG_WIDTH, OG_HEIGHT, EGO_LOWER_BOUND, EGO_UPPER_BOUND)
        raw_frame = np.array(cv2.imread(file))
        f1.set_frame_data(raw_frame)
        f1.set_class_colors(SEGMENTED_COLORS)
        f1.set_class_costs(SEGMENTATION_CLASS_COST)
        f1.process_safe_distance_zone((MIN_DISTANCE_WIDTH_PX, MIN_DISTANCE_HEIGHT_PX), False)
        return f1
        

    def test_case_1(self):
        discover = self.get_gpd()
        f1 = self.basic_test_case("bev_1.png")
        
        L1 = Waypoint(125, -27, angle.new_deg(24.0))
        L2 = Waypoint(239, -160, angle.new_deg(0.0))
        
        ego = MapPose(x=0, y=0, z=0, heading=0.0)        
        g1 = self.conv.convert(ego, L1)
        g2 = self.conv.convert(ego, L2)
        
        # 125, 52, 26.56504918231699
        
        res = discover.find(f1, ego, g1, g2, False)
        
        # self.assertEqual(res.x, 128)
        # self.assertEqual(res.z, 0)
        # self.assertEqual(res.heading, angle.new_rad(0.0))
        #discover.find()
        pass
    
if __name__ == "__main__":
    unittest.main()
