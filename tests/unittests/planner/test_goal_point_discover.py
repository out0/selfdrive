import sys, time
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.waypoint import Waypoint
from vision.occupancy_grid_cuda import OccupancyGrid, GridDirection
from data.coordinate_converter import CoordinateConverter
from planner.goal_point_discover import GoalPointDiscover, TOP, BOTTOM, LEFT, RIGHT, INSIDE
import numpy as np
import cv2
from model.physical_parameters import PhysicalParameters


class TestGoalPointDiscover(unittest.TestCase):
    
    def test_check_goal_is_in_range_and_feasible(self):

        frame = np.full((256,256,3), 1)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        inside_waypoint = Waypoint(128, 10)
        res = gpd._check_goal_is_in_range_and_feasible(og, inside_waypoint)
        self.assertTrue(res)
        
        outside_waypoint1 = Waypoint(128, -10)
        res = gpd._check_goal_is_in_range_and_feasible(og, outside_waypoint1)
        self.assertFalse(res)
        
        outside_waypoint2 = Waypoint(-20, 20)
        res = gpd._check_goal_is_in_range_and_feasible(og, outside_waypoint2)
        self.assertFalse(res)
        
        outside_waypoint3 = Waypoint(-20, -20)
        res = gpd._check_goal_is_in_range_and_feasible(og, outside_waypoint3)
        self.assertFalse(res)
    
    def test_check_too_close(self):

        frame = np.full((256,256,3), 1)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        goal1 = Waypoint(128, 0)
        res = gpd._check_too_close(og, goal1)
        self.assertFalse(res)
        
        goal2 = Waypoint(128, 110)
        res = gpd._check_too_close(og, goal2)
        self.assertTrue(res)
    
    def test_compute_direction(self):

        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        
        direction = gpd._compute_direction(Waypoint(128, 128), Waypoint(128, 0))
        self.assertEqual(direction, TOP)
        
        direction = gpd._compute_direction(Waypoint(128, 128), Waypoint(133, 0))
        self.assertEqual(direction, TOP | RIGHT)
           
        direction = gpd._compute_direction(Waypoint(128, 128), Waypoint(100, 0))
        self.assertEqual(direction, TOP | LEFT)
        
        direction = gpd._compute_direction(Waypoint(128, 0), Waypoint(128, 128))
        self.assertEqual(direction, BOTTOM)
        
        direction = gpd._compute_direction(Waypoint(128, 0), Waypoint(133, 128))
        self.assertEqual(direction, BOTTOM | RIGHT)
           
        direction = gpd._compute_direction(Waypoint(128, 0), Waypoint(100, 128))
        self.assertEqual(direction, BOTTOM | LEFT) 
        
        # PURE Left / Right are excluded in favor of TOP
        direction = gpd._compute_direction(Waypoint(128, 128), Waypoint(100, 128))
        self.assertEqual(direction, TOP | LEFT)
                    
        direction = gpd._compute_direction(Waypoint(128, 128), Waypoint(200, 128))
        self.assertEqual(direction, TOP | RIGHT)      
    
    def test_degenerate_direction(self):

        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        self.assertEqual(TOP, gpd._degenerate_direction(TOP | LEFT))
        self.assertEqual(TOP, gpd._degenerate_direction(TOP | RIGHT))
        self.assertEqual(BOTTOM, gpd._degenerate_direction(BOTTOM | LEFT))
        self.assertEqual(BOTTOM, gpd._degenerate_direction(BOTTOM | RIGHT))
        self.assertEqual(TOP, gpd._degenerate_direction(LEFT))
        self.assertEqual(TOP, gpd._degenerate_direction(RIGHT))
    

    def test_find_best_goal_in_range_X(self):

        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        frame = np.full((256, 256, 3), 1)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        

        goal = Waypoint(275, 20)  # TOP-RIGHT far goalpoint
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        ALL = int(GridDirection.ALL.value)
        
        # all pixels in TOP allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows=(0, 0),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(0, best_goal.z)
        
        # RESTRICT pixels in TOP
        og.set_goal_vectorized(goal)
        best_goal, best_cost2 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (100, 140),
                                    x_range_rows=(0, 0),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(0, best_goal.z)
        
        # RESTRICT pixels in TOP with 10 rows
        og.set_goal_vectorized(goal)
        best_goal, best_cost3 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (100, 140),
                                    x_range_rows=(0, 9),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(9, best_goal.z)
        
        self.assertGreater(best_cost2, best_cost1)
        self.assertGreater(best_cost2, best_cost3)
        
        
        # all pixels in BOTTOM allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows=(255, 255),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(255, best_goal.z)
        
        # RESTRICT pixels in BOTTOM
        og.set_goal_vectorized(goal)
        best_goal, best_cost2 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (100, 140),
                                    x_range_rows=(255, 255),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(255, best_goal.z)
        
        # RESTRICT pixels in TOP with 10 rows
        og.set_goal_vectorized(goal)
        best_goal, best_cost3 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (100, 140),
                                    x_range_rows=(100, 255),
                                    z_range = None,
                                    z_range_cols=None,
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(100, best_goal.z)
           
    def test_find_best_goal_in_range_Z(self):

        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        ALL = int(GridDirection.ALL.value)
        
        frame = np.full((256, 256, 3), 1)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        

        goal = Waypoint(275, 20)  # TOP-RIGHT far goalpoint
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        # all pixels in RIGHT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (0, 255),
                                    z_range_cols=(255, 255),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(20, best_goal.z)
        
        # RESTRICT pixels in RIGHT
        og.set_goal_vectorized(goal)
        best_goal, best_cost2 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (100, 140),
                                    z_range_cols=(255, 255),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(100, best_goal.z)
        
        # RESTRICT pixels in RIGHT with 10 rows
        og.set_goal_vectorized(goal)
        best_goal, best_cost3 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (100, 140),
                                    z_range_cols=(245, 255),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(100, best_goal.z)

         # all pixels in LEFT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (0, 255),
                                    z_range_cols=(0, 0),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(20, best_goal.z)
        
        # RESTRICT pixels in LEFT
        og.set_goal_vectorized(goal)
        best_goal, best_cost2 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (100, 140),
                                    z_range_cols=(0, 0),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(100, best_goal.z)
        
        # RESTRICT pixels in LEFT with 10 rows
        og.set_goal_vectorized(goal)
        best_goal, best_cost3 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = None,
                                    x_range_rows= None,
                                    z_range = (100, 140),
                                    z_range_cols=(0, 0),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(100, best_goal.z)
           
    def test_find_best_goal_in_range_X_Z(self):

        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        ALL = int(GridDirection.ALL.value)

        
        frame = np.full((256, 256, 3), 1)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        

        goal = Waypoint(275, 20)  # TOP-RIGHT far goalpoint
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        # all pixels in TOP-RIGHT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows = (0, 0),
                                    z_range = (0, 255),
                                    z_range_cols=(255, 255),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(20, best_goal.z)
        
        # all pixels in TOP-LEFT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost2 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows = (0, 0),
                                    z_range = (0, 255),
                                    z_range_cols=(0, 0),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(0, best_goal.z)
        
        # all pixels in BOTTOM-RIGHT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost3 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows = (255, 255),
                                    z_range = (0, 255),
                                    z_range_cols=(255, 255),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(20, best_goal.z)

        # all pixels in BOTTOM-LEFT allowed
        og.set_goal_vectorized(goal)
        best_goal, best_cost1 = gpd._find_best_goal_in_range(frame=og.get_frame(),
                                    x_range = (0, 255),
                                    x_range_rows = (255, 255),
                                    z_range = (0, 255),
                                    z_range_cols=(0, 0),
                                    relative_heading = relative_heading,
                                    allowed_directions = ALL)
        
        self.assertIsNotNone(best_goal)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(255, best_goal.z)
    
    
    def add_obstacle(self, frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]):
        
        min_z = min(p1[1], p2[1])
        max_z = max(p1[1], p2[1])
        min_x = min(p1[0], p2[0])
        max_x = max(p1[0], p2[0])
        
        for z in range (min_z, max_z + 1):
            for x in range (min_x, max_x + 1):
                frame[z, x, :] = [0, 0, 0]

    def test_find_best_goal_on_boundary_TOP_LEFT(self):
        
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)        
        frame = np.full((256, 256, 3), 1)
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        goal = Waypoint(-50, 5)  # TL
        goal2 = Waypoint(50, -50)  # T
         
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(5, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        og.set_goal_vectorized(goal2)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal2)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, 0)
        self.assertEqual(50, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        ### OBSTACLES
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (20, 60))
        
        # frame[frame[:, :, 0] == 1] = [255, 255, 255]
        # cv2.imwrite("test.png", frame)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(-50, 5)
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
    
        best_goal = gpd._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, 0)
        self.assertEqual(32, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(math.radians(-90), best_goal.heading, places=3)

        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (70, 30))
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(-50, 5)
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(45, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (130, 2))
        self.add_obstacle(frame, (0,0), (2, 130))
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(-50, 5)
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, 0)
        
        self.assertIsNone(best_goal)
        
    def test_find_best_goal_on_boundary_TOP_RIGHT(self):
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        frame = np.full((256, 256, 3), 1)
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        goal = Waypoint(300, 5)  # TR
        goal2 = Waypoint(140, -50)  # T
         
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, 0)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(5, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        og.set_goal_vectorized(goal2)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal2)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, 0)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        ### OBSTACLES
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (235, 0), (255, 60))
        

        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(300, 5)  # TR
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
    
        best_goal = gpd._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, 0)
        self.assertEqual(223, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(math.radians(-90), best_goal.heading, places=3)

        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (120,0), (255, 30))

        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(300, 5)  # TR
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, 0)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(45, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (125,0), (255, 2))
        self.add_obstacle(frame, (255,0), (255, 130))
        # frame[frame[:, :, 0] == 1] = [255, 255, 255]
        # cv2.imwrite("test.png", frame)
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        goal = Waypoint(300, 5)  # TR
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, 0)
        
        self.assertIsNone(best_goal)


    def test_find_best_goal_on_boundary_TOP(self):
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        frame = np.full((256, 256, 3), 1)
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        goal1 = Waypoint(-50, 5)  # TL
        goal2 = Waypoint(300, 5)  # TR
        goal3 = Waypoint(140, -50)  # T
         
        og.set_goal_vectorized(goal1)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal1)
        
        best_goal = gpd._find_best_goal_on_boundary_TOP(og, relative_heading, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        og.set_goal_vectorized(goal2)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal2)
        best_goal = gpd._find_best_goal_on_boundary_TOP(og, relative_heading, 0)
        self.assertEqual(255, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        og.set_goal_vectorized(goal3)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal3)
        best_goal = gpd._find_best_goal_on_boundary_TOP(og, relative_heading, 0)
        self.assertEqual(140, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        ### OBSTACLES
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0, 0), (130, 2))
        # frame[frame[:, :, 0] == 1] = [255, 255, 255]
        # cv2.imwrite("test.png", frame)
        goal = Waypoint(70, -50)  # TL
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        og.set_goal_vectorized(goal)
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal)
        best_goal = gpd._find_best_goal_on_boundary_TOP(og, relative_heading, 0)
        self.assertEqual(142, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(math.radians(-90), best_goal.heading, places=3)

    def test__find_best_goal_on_boundaries(self):
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        
        frame = np.full((256, 256, 3), 1)
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        goal1 = Waypoint(-50, 5)  # TL
        relative_heading = OccupancyGrid.compute_heading(gpd._ego_start, goal1)
         
        og.set_goal_vectorized(goal1)
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP | LEFT, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(5, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
                
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP | RIGHT, 0)
        self.assertEqual(128, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        og.set_goal_vectorized(goal1)
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP | LEFT, 0)
        self.assertEqual(0, best_goal.x)
        self.assertEqual(5, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (130, 2))
        self.add_obstacle(frame, (0,0), (2, 130))
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        og.set_goal_vectorized(goal1)
        
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP | LEFT, 0)
        self.assertEqual(142, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
        
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP, 0)
        self.assertEqual(142, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)
                
        best_goal = gpd._find_best_goal_on_boundaries(og, goal1, TOP | RIGHT, 0)
        self.assertEqual(142, best_goal.x)
        self.assertEqual(0, best_goal.z)
        self.assertAlmostEqual(relative_heading, best_goal.heading, places=3)

    def test_find_any_feasible_in_direction(self):
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        ALL = int(GridDirection.ALL.value)

        
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (2, 150))
        self.add_obstacle(frame, (255,0), (255, 150))
        self.add_obstacle(frame, (0,0), (255, 2))
        
        # frame[frame[:, :, 0] == 1] = [255, 255, 255]
        # cv2.imwrite("test.png", frame)
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        goal1 = Waypoint(-50, 5)  # TL
        og.set_goal_vectorized(goal1)
        best_goal = gpd._find_any_feasible_in_direction(og, goal1, TOP)
        self.assertEqual(14, best_goal.x)
        self.assertEqual(17, best_goal.z)
        self.assertAlmostEqual(math.radians(-90), best_goal.heading, places=3)

    

    def test_find_local_goal_for_out_of_range_goal(self):
        converter = CoordinateConverter(world_origin=WorldPose(0,0,0,0))
        gpd = GoalPointDiscover(converter)
        gpd._ego_start = Waypoint(128, 128)
        ALL = int(GridDirection.ALL.value)

        
        frame = np.full((256, 256, 3), 1)
        self.add_obstacle(frame, (0,0), (2, 150))
        self.add_obstacle(frame, (255,0), (255, 150))
        self.add_obstacle(frame, (0,0), (255, 2))
        
        goal1 = Waypoint(-20, 5)  # TL
        dist = Waypoint.distance_between(gpd._ego_start, goal1)
        self.assertLess(dist, 200, "wrong test case! I dont want it to degenerate")
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        best_goal = gpd._find_local_goal_for_out_of_range_goal(og, goal1)
        self.assertEqual(14, best_goal.goal.x)
        self.assertEqual(17, best_goal.goal.z)
        
        ## DEGENERATING
        frame = np.full((256, 256, 3), 1)
        goal1 = Waypoint(-50, 5)  # TL
        dist = Waypoint.distance_between(gpd._ego_start, goal1)
        self.assertGreater(dist, 200, "wrong test case! I want it to degenerate")
        
        og = OccupancyGrid(
            frame, 
            PhysicalParameters.MIN_DISTANCE_WIDTH_PX, 
            PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            PhysicalParameters.EGO_LOWER_BOUND,
            PhysicalParameters.EGO_UPPER_BOUND
        )
        
        best_goal = gpd._find_local_goal_for_out_of_range_goal(og, goal1)
        self.assertEqual(0, best_goal.goal.x)
        self.assertEqual(0, best_goal.goal.z)

if __name__ == "__main__":
    unittest.main()


