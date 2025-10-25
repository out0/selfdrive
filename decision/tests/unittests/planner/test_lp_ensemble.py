import sys, time
sys.path.append("../../../")
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import SearchFrame, EgoParams, SearchParams
from pydriveless import CoordinateConverter
import unittest
import numpy as np
from ensemble.planner.ensemble import Ensemble
from ensemble.model.planning_data import PlanningData
from ensemble.model.planning_result import PlanningResult, PlannerResultType
from ensemble.model.physical_paramaters import PhysicalParameters
import cv2
import cProfile, timeit

class TestLPEnsemble(unittest.TestCase):

    TIMEOUT_MS = -1
    ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

    def test_free_area(self):
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

               
        planner = Ensemble(ego_params)        
        planner.plan(search_params, True)
       
        self.assertTrue(planner.get_execution_time() > 0)
        
        result = planner.get_result()
        
        self.assertEqual(result.result_type, PlannerResultType.VALID)
        
        # for p in result.path:
        #     if (p.x > 52 or p.x < 48):
        #         self.fail("should be straight or near straight line")
        
        planner.cancel()
        while planner.is_running():
            pass

        f = frame.get_color_frame()
        if result.path is not None:
            for p in result.path:
                f[p.z, p.x] = [0, 255, 0]
        
        cv2.imwrite("debug.png", f)

        print(str(result))
       
    def test_diverge_plan_due_to_obstacle(self):
        ego_params = EgoParams.init(100, 100)\
                .with_max_steering_angle(angle.new_deg(40))\
                .with_max_curvature(0.34)\
                .with_segmentation_class_costs(np.array([0.0, -1.0]))\
                .with_segmentation_class_colors(np.array([[255, 255, 255], [0, 0, 0]]))\
                .with_search_physical_size(1, 1)\
                .build()

        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        for z in range(0, 10):
            for x in range(0, 100):
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
            .with_timeout(100)\
            .build()

               
        planner = Ensemble(ego_params)        
        planner.plan(search_params, True)
       
        self.assertTrue(planner.get_execution_time() > 0)
        self.assertFalse(planner.new_path_available())
        
        result = planner.get_result()       
        self.assertIsNone(result)

    def test_bev_1(self):
        bev = np.array(cv2.imread("bev_1.png"), dtype=np.float32)
        ego_params = EgoParams.init(256, 256)\
                .with_max_steering_angle(angle.new_deg(PhysicalParameters.MAX_STEERING_ANGLE))\
                .with_ego_lower_bound(PhysicalParameters.EGO_LOWER_BOUND)\
                .with_ego_upper_bound(PhysicalParameters.EGO_UPPER_BOUND)\
                .with_max_curvature(0.34)\
                .with_segmentation_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)\
                .with_segmentation_class_colors(PhysicalParameters.SEGMENTED_COLORS)\
                .with_search_physical_size(PhysicalParameters.OG_REAL_WIDTH, PhysicalParameters.OG_REAL_HEIGHT)\
                .build()

        goal = Waypoint(x=108, z=0, heading=angle.new_deg(0.0))

        frame = ego_params.new_search_frame()
        frame.set_frame_data(bev)
        frame.process_distance_to_goal(goal.x, goal.z)
        frame.process_safe_distance_zone((PhysicalParameters.MIN_DISTANCE_WIDTH_PX//2, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX//2), True)

        search_params = ego_params.new_search_params(goal=goal)\
            .with_distance_to_goal_tolerance(20.0)\
            .with_frame(frame)\
            .with_max_path_size(40.0)\
            .with_min_distance((PhysicalParameters.MIN_DISTANCE_WIDTH_PX//2, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX//2))\
            .with_velocity(1.0)\
            .with_distance_to_goal_tolerance(5)\
            .with_timeout(500)\
            .build()

               
        planner = Ensemble(ego_params)        
        planner.plan(search_params, True)

        self.assertTrue(planner.get_execution_time() > 0)
        
        result = planner.get_result()
        
        self.assertEqual(result.result_type, PlannerResultType.VALID)
        
        f = frame.get_color_frame()
        if result.path is not None:
            for p in result.path:
                f[p.z, p.x] = [0, 255, 0]
        
        cv2.imwrite("debug.png", f)
        
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))

if __name__ == "__main__":
    unittest.main()
