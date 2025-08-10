import sys, time
sys.path.append("../../../../")
sys.path.append("../../../")
from ensemble.planner.debug import Debug
from ensemble.model.physical_paramaters import PhysicalParameters
import unittest
import cv2, numpy as np
from pydriveless import SearchFrame

class TestFeasiblePath(unittest.TestCase):
        
    def test_overtaker_direct_path_bug(self):
        path = Debug.read_path("model/direct.log")

        frame = np.array(cv2.imread("bev_1.png"))
        og = SearchFrame(
            width=frame.shape[1],
            height=frame.shape[0],
            lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
            upper_bound=PhysicalParameters.EGO_UPPER_BOUND
        )
        og.set_frame_data(frame)
        og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        og.process_safe_distance_zone((PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX), False)

        res = og.check_feasible_path((PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX), path)
        self.assertTrue(res)




if __name__ == "__main__":
    unittest.main()
