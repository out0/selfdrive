import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math, numpy as np

from model.planning_data import PrePlanningData, PlanningData
from model.map_pose import MapPose

class TestPrePlanningData(unittest.TestCase):

    def test_pre_planning_data_to_str(self):
        
        planning = PrePlanningData(
            pose=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            top=np.zeros((100,100,2)),
            left=np.zeros((110,100,3)),
            right=np.zeros((100,110,2)),
            bottom=None
        )
        p = str(planning)
        self.assertEqual(p, f"pose:{planning.pose},velocity:11.2,top:(100,100,2),left:(110,100,3),right:(100,110,2),bottom:()")
        
    def test_pre_planning_data_init_params(self):
        planning = PrePlanningData(
            pose=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            top=np.zeros((100,100,2)),
            left=np.zeros((110,100,3)),
            right=np.zeros((100,110,2)),
            bottom=np.zeros((100,110,4)),
        )
        self.assertEqual(planning.top_frame.shape, (100, 100, 2))
        self.assertEqual(planning.left_frame.shape, (110, 100, 3))
        self.assertEqual(planning.right_frame.shape, (100, 110, 2))
        self.assertEqual(planning.bottom_frame.shape, (100, 110, 4))
        self.assertAlmostEqual(planning.velocity, 11.2, 4)
        self.assertAlmostEqual(str(planning.pose), str(MapPose(1.1, 2.2, 3.3, 10.1)))
        
class TestPlanningData(unittest.TestCase):

    def test_planning_data_to_str(self):
        
        planning = PlanningData(
            pose=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            bev=np.zeros((100,100,2)),
        )
        p = str(planning)
        self.assertEqual(p, f"pose:{planning.pose},velocity:11.2,bev:(100,100,2)")

    def test_planning_data_init_params(self):
        planning = PlanningData(
            pose=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            bev=np.zeros((100,100,2)),
        )
        self.assertEqual(planning.bev.shape, (100, 100, 2))
        self.assertAlmostEqual(planning.velocity, 11.2, 4)
        self.assertAlmostEqual(str(planning.pose), str(MapPose(1.1, 2.2, 3.3, 10.1)))

if __name__ == "__main__":
    unittest.main()
