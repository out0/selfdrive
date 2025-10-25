import sys, time
sys.path.append("../../../")
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import SearchFrame, EgoParams, SearchParams
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

        ego_params = EgoParams.init(100, 100)\
                .with_max_steering_angle(angle.new_deg(40))\
                .with_max_curvature(0.34)\
                .with_segmentation_class_costs(np.array([0.0, -1.0]))\
                .with_segmentation_class_colors(np.array([[255, 255, 255], [0, 0, 0]]))\
                .with_search_physical_size(1, 1)\
                .build()

        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        goal = Waypoint(x=50, z=0, heading=angle.new_deg(0.0))

        frame = ego_params.new_search_frame()
        frame.set_frame_data(bev)
        frame.process_distance_to_goal(goal.x, goal.z)
        frame.process_safe_distance_zone((5,5), True)

        search_params = ego_params.new_search_params(goal=goal)\
            .with_distance_to_goal_tolerance(20.0)\
            .with_frame(frame)\
            .with_max_path_size(40.0)\
            .with_min_distance((5,5))\
            .with_velocity(1.0)\
            .with_distance_to_goal_tolerance(5)\
            .with_timeout(3000)\
            .build()

               
        planner = Interpolator(ego_params)        
        planner.plan(search_params, True)
       
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
        ego_params = EgoParams.init(100, 100)\
                .with_max_steering_angle(angle.new_deg(40))\
                .with_max_curvature(0.34)\
                .with_segmentation_class_costs(np.array([0.0, -1.0]))\
                .with_segmentation_class_colors(np.array([[255, 255, 255], [0, 0, 0]]))\
                .with_search_physical_size(1, 1)\
                .build()

        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        for z in range(0, 10):
            for x in range(40, 60):
                bev[z,x,0] = 1.0

        goal = Waypoint(x=50, z=0, heading=angle.new_deg(0.0))

        frame = ego_params.new_search_frame()
        frame.set_frame_data(bev)
        frame.process_distance_to_goal(goal.x, goal.z)
        frame.process_safe_distance_zone((5,5), True)

        search_params = ego_params.new_search_params(goal=goal)\
            .with_distance_to_goal_tolerance(20.0)\
            .with_frame(frame)\
            .with_max_path_size(40.0)\
            .with_min_distance((5,5))\
            .with_velocity(1.0)\
            .with_distance_to_goal_tolerance(5)\
            .with_timeout(3000)\
            .build()

               
        planner = Interpolator(ego_params)        
        planner.plan(search_params, True)
       
        self.assertTrue(planner.get_execution_time() > 0)
        
        result = planner.get_result()
        
        self.assertEqual(result.result_type, PlannerResultType.INVALID_PATH)
        
        for p in result.path:
            if (p.x > 52 or p.x < 48):
                self.fail("should be straight or near straight line")
        
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))


       
        

        

if __name__ == "__main__":
    unittest.main()
