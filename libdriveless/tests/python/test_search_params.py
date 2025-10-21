import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import SearchFrame, angle, Waypoint
from pydriveless import EgoParams, SearchParams

class TestSearchParams(unittest.TestCase):
        
    def test_ego_params_default(self):
        params = EgoParams.init(256, 256).build()

        self.assertTupleEqual(params.ego_lower_bound, (-1, -1))
        self.assertTupleEqual(params.ego_upper_bound, (-1, -1))

        self.assertEqual(params.max_steering_angle, angle.new_deg(40))
        self.assertEqual(params.max_curvature, 0.35)
        self.assertEqual(params.pixel_to_meter_ratio_width, 1.0)
        self.assertEqual(params.pixel_to_meter_ratio_height, 1.0)
        self.assertEqual(params.meter_to_pixel_ratio_width, 1.0)
        self.assertEqual(params.meter_to_pixel_ratio_height, 1.0)
 
    def test_search_params_default(self):
        params = EgoParams.init(256, 256).build()

        goal = Waypoint(128, 0, angle.new_deg(11))
        search: SearchParams = params.new_search_params(goal=goal).build()

        self.assertEqual(search.timeout_ms, 350)
        self.assertAlmostEqual(search.max_path_size_px, 30.0)
        self.assertAlmostEqual(search.distance_to_goal_tolerance_px, 20.0)
        self.assertEqual(search.heading_error_tolerance, angle.new_deg(5))
        self.assertTupleEqual(search.min_distance, (0, 0))
        self.assertEqual(search.frame, None)
        self.assertTrue(search.start == Waypoint(128, 128, angle.new_deg(0)))
        self.assertTrue(search.goal == Waypoint(128, 0, angle.new_deg(11)))
        self.assertAlmostEqual(search.velocity_m_s, 1.0)

    def test_build_ego_params_custom_values(self):
            params = (EgoParams.init(256, 256)
                    .with_ego_lower_bound((10, 11))
                    .with_ego_upper_bound((12, 13))
                    .with_max_curvature(2.0)
                    .with_max_steering_angle(angle.new_deg(7))
                    .with_search_physical_size(25.6, 25.6)
                    .with_segmentation_class_colors([(0, 0, 0), (255, 255, 255)])
                    .with_segmentation_class_costs([0.0, -1.0])
                    .with_vehicle_length(3.2)
                    .build())

            self.assertTupleEqual(params.ego_lower_bound, (10, 11))
            self.assertTupleEqual(params.ego_upper_bound, (12, 13))

            self.assertEqual(params.max_curvature, 2.0)
            self.assertEqual(params.max_steering_angle, angle.new_deg(7))
            self.assertAlmostEqual(params.vehicle_length_m, 3.2)
            self.assertAlmostEqual(params.pixel_to_meter_ratio_width, 0.1)
            self.assertAlmostEqual(params.pixel_to_meter_ratio_height, 0.1)
            self.assertAlmostEqual(params.meter_to_pixel_ratio_width, 10.0)
            self.assertAlmostEqual(params.meter_to_pixel_ratio_height, 10.0)

            colors = [list(c) for c in params.segmentation_class_colors]
            self.assertEqual(colors[0], [0, 0, 0])
            self.assertEqual(colors[1], [255, 255, 255])

            costs = params.segmentation_class_costs
            self.assertAlmostEqual(costs[0], 0.0)
            self.assertAlmostEqual(costs[1], -1.0)
 
    def test_build_search_params_custom_values(self):
            ego = (EgoParams.init(256, 256)
                .with_ego_lower_bound((10, 11))
                .with_ego_upper_bound((12, 13))
                .with_max_curvature(2.0)
                .with_max_steering_angle(angle.new_deg(7))
                .with_search_physical_size(25.6, 25.6)
                .with_segmentation_class_colors([(0, 0, 0), (255, 255, 255)])
                .with_segmentation_class_costs([0.0, -1.0])
                .with_vehicle_length(3.2)
                .build())

            frame = ego.new_search_frame()

            search = (ego.new_search_params(
                        start=Waypoint(128, 107, angle.new_deg(-6.3)),
                        goal=Waypoint(128, 5, angle.new_deg(11)))
                    .with_distance_to_goal_tolerance(10.12)
                    .with_frame(frame)
                    .with_heading_error_tolerance(angle.new_deg(12.3))
                    .with_max_path_size(1.234)
                    .with_min_distance((10.12, 11.14))
                    .with_timeout(501)
                    .with_velocity(3.45)
                    .build())

            self.assertEqual(search.timeout_ms, 501)
            self.assertAlmostEqual(search.max_path_size_px, 1.234)
            self.assertAlmostEqual(search.distance_to_goal_tolerance_px, 10.12)
            self.assertEqual(search.heading_error_tolerance, angle.new_deg(12.3))
            self.assertTupleEqual(search.min_distance, (10.12, 11.14))
            self.assertIs(search.frame, frame)
            self.assertEqual(search.start, Waypoint(128, 107, angle.new_deg(-6.3)))
            self.assertEqual(search.goal, Waypoint(128, 5, angle.new_deg(11)))
            self.assertAlmostEqual(search.velocity_m_s, 3.45)


if __name__ == "__main__":
    unittest.main()

