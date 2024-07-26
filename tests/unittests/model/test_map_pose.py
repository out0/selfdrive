import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from model.map_pose import MapPose


class TestMapPose(unittest.TestCase):

    def test_map_pose_encode_decode(self):
        pose1 = MapPose(10, -10.2, 23.3456, 10.2)
        p = str(pose1)
        pose2 = MapPose.from_str(p)
        
        self.assertEqual(pose1.x, pose2.x)
        self.assertEqual(pose1.y, pose2.y)
        self.assertEqual(pose1.z, pose2.z)
        self.assertEqual(pose1.heading, pose2.heading)
        

    def test_map_pose_distance_between(self):
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,0,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 10, 4)
        
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 10 * math.sqrt(2), 4)
        
        p1 = MapPose(-10,-10,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 20 * math.sqrt(2), 4)
        
        p1 = MapPose(10,10,0,0)
        p2 = MapPose(10,10,0,0)       
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 0, 4)

    def test_map_pose_dot(self):
        p1 = MapPose(0,0,0,0)
        p2 = MapPose(10,10,0,0)
        self.assertAlmostEqual(MapPose.dot(p1, p2), 0, 4)

        p1 = MapPose(10,10,0,0)
        p2 = MapPose(10,10,0,0)
        self.assertAlmostEqual(MapPose.dot(p1, p2), 200, 4)

    def test_distance_to_line(self):
        line_p1 = MapPose(2, 2, 0, heading=0)
        line_p2 = MapPose(6, 6, 0, heading=0)
        p = MapPose(4, 0, 0, heading=0)
        
        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 2*math.sqrt(2), places=3)
        
        p = MapPose(6, 0, 0, heading=0)
        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 3*math.sqrt(2), places=3)

    def test_compute_path_heading(self):
        line_p1 = MapPose(2, 2, 0, heading=0)
        line_p2 = MapPose(6, 6, 0, heading=0)
                
        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertAlmostEqual(heading, math.pi/4, places=3)
        
        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertAlmostEqual(heading, -math.pi/4 - math.pi/2, places=3)
        
        line_p1 = MapPose(0, 0, 0, heading=0)
        line_p2 = MapPose(0, 6, 0, heading=0)
        
        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertAlmostEqual(heading, math.pi/2, places=3)
        
        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertAlmostEqual(heading, -math.pi/2, places=3)
    
    def test_equality_operation(self):
        p1 = MapPose(2, 2, 0, heading=0)
        p2 = MapPose(2, 2, 0, heading=0)
        
        self.assertTrue(p1 == p2)
        
        p2.heading = 1       
        self.assertFalse(p1 == p2)
        
        p2.heading = 0
        p2.x += 1
        self.assertFalse(p1 == p2)
        
        p2.x -= 1
        p2.y += 1
        self.assertFalse(p1 == p2)

        p2.y -= 1
        p2.z += 1
        self.assertFalse(p1 == p2)

        
        class ExtendedMapose(MapPose):
            def __init__(self, x: float, y: float, z: float, heading: float):
                super().__init__(x, y, z, heading)

        ext = ExtendedMapose(2, 2, 0, heading=0)
        self.assertTrue(p1 == ext)
        
    
    def test_project_on_path(self):
        line_p1 = MapPose(2, 2, 0, heading=0)
        line_p2 = MapPose(6, 6, 0, heading=0)
        p = MapPose(4, 0, 0, heading=0)
        
        projected, dist, path_size = MapPose.project_on_path(line_p1, line_p2, p)
        
        self.assertAlmostEqual(dist, 0, places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(2, 2, 0, heading=0), projected)
        
        p = MapPose(6, 0, 0, heading=0)
        
        projected, dist, path_size = MapPose.project_on_path(line_p1, line_p2, p)
        
        self.assertAlmostEqual(dist,  math.sqrt(2), places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(3, 3, 0, heading=0), projected)


    def plot_segments_and_points(self, segments, points, expected_pos):
        _, ax = plt.subplots()

        # Plot segments
        x, y = segments[:, 0], segments[:, 1]
        ax.plot(x, y, color='blue')
        ax.scatter(x, y, color='blue', marker='X')

        # Plot points
        x, y = points[:, 0], points[:, 1]
        ax.scatter(x, y, color='red', marker='o')

        
        x = np.zeros((len(expected_pos)), dtype=np.int32)
        y = np.zeros((len(expected_pos)), dtype=np.int32)

        i = 0
        for p in expected_pos:
            x[i] = segments[p, 0]
            y[i] = segments[p, 1]
            i += 1

        ax.scatter(x, y, color='yellow', marker='X')
        plt.savefig("test_find_nearest_goal_pose.png")


    def test_find_nearest_goal_pose(self):
        segments = np.array([
            [0, 0],
            [10,0],
            [20,0],
            [25,5],
            [30,10],
            [35,18],
            [35,25],
            [35,35],
            [30,41],
            [25,43],
            [20,43],
            [10,43],
            [5,33],
            [0,23],
            [0,13],
        ])
        
        path = []
        for p in segments:
            path.append(MapPose(x=p[0], y=p[1], z=0, heading=0))

        points = np.array([
            [0, 0],
            [5, 2],
            [10, -1],
            [17, -3],
            [23, -4],
            [24, 6],
            [29, 8],
            [29, 17],
            [35,18],
            [35,20],
            [35,22],
            [35,24],
            [28,34],
            [28,39],
            [20,39],
            [18,39],
            [14,39],
            [14,29],
            [5, 19],    
            [5, 15],  
            [5, 12],
        ])

        expected_pos = np.array([
            1,
            1,
            2,
            3,
            3,
            4,
            4,
            5,
            6,
            6,
            6,
            7,
            8,
            9,
            11,
            11,
            11, #16
            13,
            14,
            -1
        ], dtype=np.int32)
        
        for i in range (len(expected_pos)):
            if i == 18:
                pass
            location = MapPose(x=points[i, 0], y=points[i, 1], z=0, heading=0)
            expected_pos_for_location = expected_pos[i]            
            obtained_pos_for_location = MapPose.find_nearest_goal_pose(location, path, 0)
            

            
            self.assertEqual(expected_pos_for_location, obtained_pos_for_location, 
                             f"wrong goal pose: expected: {expected_pos_for_location} ({points[expected_pos_for_location]}) obtained: {obtained_pos_for_location} ({points[obtained_pos_for_location]}) for test #{i}")


        self.plot_segments_and_points(segments, points, expected_pos)

if __name__ == "__main__":
    unittest.main()
