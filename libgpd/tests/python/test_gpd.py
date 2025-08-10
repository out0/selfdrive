import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import angle
from pydriveless import MapPose, WorldPose, Waypoint
from pydriveless import CoordinateConverter
from pygpd import GoalPointDiscover
from pydriveless import SearchFrame


import cv2
from test_utils import fix_cv2_import
fix_cv2_import()

OG_WIDTH = 256
OG_HEIGHT = 256
OG_REAL_WIDTH: float = 34.641016151377535
OG_REAL_HEIGHT: float = 34.641016151377535

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

    def test_in_range_no_obstacle(self):
        discover = self.get_gpd()
        
        f1 = SearchFrame(256, 256, (5, 5), (15, 15))
        raw_frame = np.full((256, 256, 3), fill_value=1.0, dtype=np.float32)
        f1.set_frame_data(raw_frame)
        
        # Set class colors: class 0 -> white, class 1 -> black
        f1.set_class_colors(np.array([(255, 255, 255), (0, 0, 0)]))

        # Set class costs: class 0 -> -1, class 1 -> 0
        f1.set_class_costs(np.array([-1, 0]))

        # Process safe distance zone at (10, 10), do not mark as obstacle
        f1.process_safe_distance_zone((10, 10), False)
        
        ego = MapPose(x=0, y=0, z=0, heading=0.0)
        
        g1 = self.conv.convert(ego, Waypoint(128, 0, angle.new_rad(0.0)))
        g2 = self.conv.convert(ego, Waypoint(128, -100, angle.new_rad(0.0)))
        
        res = discover.find(f1, ego, g1, g2, False)
        
        self.assertEqual(res.x, 128)
        self.assertEqual(res.z, 0)
        self.assertEqual(res.heading, angle.new_rad(0.0))
        #discover.find()
        pass
    
if __name__ == "__main__":
    unittest.main()
