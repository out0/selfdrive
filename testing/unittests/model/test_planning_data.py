import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math, numpy as np

from model.planning_data import PrePlanningData, PlanningData, PlanningResult, PlannerResultType
from model.map_pose import MapPose
from model.physical_parameters import PhysicalParameters
from model.waypoint import Waypoint

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


class TestPlanningResult(unittest.TestCase):
    
    def test_planning_result_str_codec(self):
        res = PlanningResult(
            ego_location = MapPose(
                x=1.1,
                y=1.2,
                z=1.3,
                heading=1.4
            ),
            direction = 2,
            local_goal = Waypoint(
                x=10,
                z=20,
                heading=30
            ),
            local_start = Waypoint(
                x=1,
                z=2,
                heading=3
            ),
            goal = MapPose(
                x=2.1,
                y=2.2,
                z=2.3,
                heading=1.4
            ),
            next_goal = MapPose(
                x=22.1,
                y=22.2,
                z=22.3,
                heading=22.4
            ),
            planner_name = "name1",
            result_type = PlannerResultType.TOO_CLOSE,
            timeout = True,
            path = [
                Waypoint(0, 0, 0),
                Waypoint(0, 0, 1),
                Waypoint(0, 1, 0),
                Waypoint(1, 0, 0),
            ],
            total_exec_time_ms=10
        )
        
        str_val = str(res)
        
        res2 = PlanningResult.from_str(str_val)
        
        self.assertEqual(res.result_type, res2.result_type)
        self.assertEqual(res.path, res2.path)
        self.assertEqual(res.timeout, res2.timeout)
        self.assertEqual(res.planner_name, res2.planner_name)
        self.assertEqual(res.total_exec_time_ms, res2.total_exec_time_ms)
        self.assertEqual(res.local_start, res2.local_start)
        self.assertEqual(res.local_goal, res2.local_goal)
        self.assertEqual(res.goal_direction, res2.goal_direction)
        self.assertEqual(res.ego_location, res2.ego_location)
        self.assertEqual(res.map_goal, res2.map_goal)
        self.assertEqual(res.map_next_goal, res2.map_next_goal)


if __name__ == "__main__":
    unittest.main()
