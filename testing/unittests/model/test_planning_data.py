import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math, numpy as np

from model.planning_data import PrePlanningData, PlanningData
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters

class TestPrePlanningData(unittest.TestCase):

    def test_pre_planning_data_to_str(self):
        
        planning = PrePlanningData(
            ego_location=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            top=np.zeros((100,100,2)),
            left=np.zeros((110,100,3)),
            right=np.zeros((100,110,2)),
            bottom=None
        )
        p = str(planning)
        self.assertEqual(p, f"ego_location:{planning.ego_location},velocity:11.2,top:(100,100,2),left:(110,100,3),right:(100,110,2),bottom:()")
        
    def test_pre_planning_data_init_params(self):
        planning = PrePlanningData(
            ego_location=MapPose(1.1, 2.2, 3.3, 10.1),
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
        self.assertAlmostEqual(str(planning.ego_location), str(MapPose(1.1, 2.2, 3.3, 10.1)))
        
class TestPlanningData(unittest.TestCase):

    def test_planning_data_to_str(self):
        
        planning = PlanningData(
            ego_location=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            bev=np.zeros((100,100,2)),
            goal=MapPose(3.3, 4.4, 5.5, 6.6),
            next_goal=MapPose(7.7, 8.8, 9.9, 10.1)
        )
        p = str(planning)
        self.assertEqual(p, f"ego_location:{planning.ego_location},velocity:11.2,bev:(100,100,2),goal:{planning.goal},next_goal:{planning.next_goal}")

    def test_planning_data_init_params(self):
        planning = PlanningData(
            ego_location=MapPose(1.1, 2.2, 3.3, 10.1),
            velocity=11.2,
            bev=np.zeros((100,100,2)),
            goal=MapPose(3.3, 4.4, 5.5, 6.6),
            next_goal=MapPose(7.7, 8.8, 9.9, 10.1)
        )
        self.assertEqual(planning.bev.shape, (100, 100, 2))
        self.assertAlmostEqual(planning.velocity, 11.2, 4)
        self.assertAlmostEqual(str(planning.ego_location), str(MapPose(1.1, 2.2, 3.3, 10.1)))
        self.assertAlmostEqual(str(planning.goal), str(MapPose(3.3, 4.4, 5.5, 6.6)))
        self.assertAlmostEqual(str(planning.next_goal), str(MapPose(7.7, 8.8, 9.9, 10.1)))
        self.assertIsNotNone(planning.og)
        
        self.assertEqual(planning.og.width(), 100)
        self.assertEqual(planning.og.height(), 100)

if __name__ == "__main__":
    unittest.main()
