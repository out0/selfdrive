import sys, time
sys.path.append("../../../")
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import SearchFrame
from pydriveless import CoordinateConverter
import unittest
import numpy as np
from ensemble.planner.interpolator import Interpolator
from ensemble.model.planning_data import PlanningData
from ensemble.model.planning_result import PlanningResult, PlannerResultType

class TestLPInterpolator(unittest.TestCase):

    TIMEOUT_MS = -1
    ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

    def test_free_area_interpolate(self):
        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        og = SearchFrame(width=100, height=100, lower_bound=(-1, -1), upper_bound=(-1, -1))
        og.set_frame_data(bev)
        og.set_class_costs(np.array([0.0, -1.0]))
        
        conv = CoordinateConverter(origin=TestLPInterpolator.ORIGIN, width=100, height=100, perceptionHeightSize_m=1, perceptionWidthSize_m=1)
        planner = Interpolator(conv, max_exec_time_ms=TestLPInterpolator.TIMEOUT_MS)
        
        ego_location = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        g1 = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        L2 = Waypoint(x=50, z=-100, heading=angle.new_rad(0))
        g2: MapPose = conv.convert(ego_location, L2)
        
        planning_data = PlanningData(
            seq=0,
            og=og,
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(5, 5)
        )
        
        planning_data.set_local_goal(Waypoint(x=50, z=0, heading=angle.new_deg(0.0)))
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        
        self.assertTrue(planner.get_execution_time() > 0)
        
        result = planner.get_result()
        
        self.assertEqual(result.result_type, PlannerResultType.VALID)
        
        for p in result.path:
            if (p.x > 52 or p.x < 48):
                self.fail("should be straight or near straight line")
        
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))
       
    def test_no_plan_due_to_obstacle(self):
        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        og = SearchFrame(width=100, height=100, lower_bound=(-1, -1), upper_bound=(-1, -1))
        
        for z in range(0, 10):
            for x in range(40, 60):
                bev[z,x,0] = 1.0
        
        og.set_frame_data(bev)
        og.set_class_costs(np.array([0.0, -1.0]))
        
        conv = CoordinateConverter(origin=TestLPInterpolator.ORIGIN, width=100, height=100, perceptionHeightSize_m=1, perceptionWidthSize_m=1)
        planner = Interpolator(conv, max_exec_time_ms=TestLPInterpolator.TIMEOUT_MS)
        
        ego_location = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        g1 = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        L2 = Waypoint(x=50, z=-100, heading=angle.new_rad(0))
        g2: MapPose = conv.convert(ego_location, L2)
        
        planning_data = PlanningData(
            seq=0,
            og=og,
            start=Waypoint(128,128, heading=angle.new_rad(0)),
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(5, 5)
        )
        
        planning_data.set_local_goal(Waypoint(x=50, z=0, heading=angle.new_deg(0.0)))
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        result = planner.get_result()

        self.assertEqual(result.result_type, PlannerResultType.INVALID_PATH)
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))
        

        

if __name__ == "__main__":
    unittest.main()
