import sys
sys.path.append("../../../")
import numpy as np
import math
import unittest
from pydriveless import MapPose, angle
import matplotlib.pyplot as plt
import time


class TestMapPose(unittest.TestCase):

    def test_map_pose_encode_decode(self):
        pose1 = MapPose(10, -10.2, 23.3456, angle.new_deg(10.2))
        p = str(pose1)
        pose2 = MapPose.from_str(p)

        self.assertEqual(pose1.x, pose2.x)
        self.assertEqual(pose1.y, pose2.y)
        self.assertEqual(pose1.z, pose2.z)
        self.assertEqual(pose1.heading, pose2.heading)

    def test_map_pose_distance_between(self):
        p1 = MapPose(0, 0, 0, angle.new_deg(0))
        p2 = MapPose(10, 0, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 10, 4)

        p1 = MapPose(0, 0, 0, angle.new_deg(0))
        p2 = MapPose(10, 10, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.distance_between(
            p1, p2), 10 * math.sqrt(2), 4)

        p1 = MapPose(-10, -10, 0, angle.new_deg(0))
        p2 = MapPose(10, 10, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.distance_between(
            p1, p2), 20 * math.sqrt(2), 4)

        p1 = MapPose(10, 10, 0, angle.new_deg(0))
        p2 = MapPose(10, 10, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.distance_between(p1, p2), 0, 4)

    def test_map_pose_dot(self):
        p1 = MapPose(0, 0, 0, angle.new_deg(0))
        p2 = MapPose(10, 10, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.dot(p1, p2), 0, 4)

        p1 = MapPose(10, 10, 0, angle.new_deg(0))
        p2 = MapPose(10, 10, 0, angle.new_deg(0))
        self.assertAlmostEqual(MapPose.dot(p1, p2), 200, 4)

    def test_distance_to_line(self):
        line_p1 = MapPose(2, 2, 0, heading=angle.new_deg(0))
        line_p2 = MapPose(6, 6, 0, heading=angle.new_deg(0))
        p = MapPose(4, 0, 0, heading=angle.new_deg(0))

        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 2*math.sqrt(2), places=3)

        p = MapPose(6, 0, 0, heading=angle.new_deg(0))
        dist = MapPose.distance_to_line(line_p1, line_p2, p)
        self.assertAlmostEqual(dist, 3*math.sqrt(2), places=3)

    def test_compute_path_heading(self):
        line_p1 = MapPose(2, 2, 0, heading=angle.new_deg(0))
        line_p2 = MapPose(6, 6, 0, heading=angle.new_deg(0))

        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertEqual(heading, angle.new_rad(math.pi/4))

        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertEqual(heading, angle.new_rad(-math.pi/4 - math.pi/2))

        line_p1 = MapPose(0, 0, 0, heading=angle.new_deg(0))
        line_p2 = MapPose(0, 6, 0, heading=angle.new_deg(0))

        heading = MapPose.compute_path_heading(line_p1, line_p2)
        self.assertEqual(heading, angle.new_rad(math.pi/2))

        heading = MapPose.compute_path_heading(line_p2, line_p1)
        self.assertEqual(heading, angle.new_rad(-math.pi/2))

    def test_equality_operation(self):
        p1 = MapPose(2, 2, 0, heading=angle.new_deg(0))
        p2 = MapPose(2, 2, 0, heading=angle.new_deg(0))

        self.assertTrue(p1 == p2)

        p2 = MapPose(2, 2, 0, heading=angle.new_deg(1))
        self.assertFalse(p1 == p2)

        MapPose(3, 2, 0, heading=angle.new_deg(0))
        self.assertFalse(p1 == p2)

        MapPose(2, 3, 0, heading=angle.new_deg(0))
        self.assertFalse(p1 == p2)

        MapPose(2, 2, 1, heading=angle.new_deg(0))
        self.assertFalse(p1 == p2)

        class ExtendedMapose(MapPose):
            def __init__(self, x: float, y: float, z: float, heading: float):
                super().__init__(x, y, z, heading)

        ext = ExtendedMapose(2, 2, 0, heading=angle.new_deg(0))
        self.assertTrue(p1 == ext)

    def test_project_on_path(self):
        line_p1 = MapPose(2, 2, 0, heading=angle.new_deg(0))
        line_p2 = MapPose(6, 6, 0, heading=angle.new_deg(0))
        p = MapPose(4, 0, 0, heading=angle.new_deg(0))

        projected, dist, path_size = MapPose.project_on_path(
            line_p1, line_p2, p)

        self.assertAlmostEqual(dist, 0, places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(2, 2, 0, heading=angle.new_deg(0)), projected)

        p = MapPose(6, 0, 0, heading=angle.new_deg(0))

        projected, dist, path_size = MapPose.project_on_path(
            line_p1, line_p2, p)

        self.assertAlmostEqual(dist,  math.sqrt(2), places=3)
        self.assertAlmostEqual(path_size, math.sqrt(4**2 + 4**2), places=3)
        self.assertEqual(MapPose(3, 3, 0, heading=angle.new_deg(0)), projected)

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

        ax.scatter(x, y, color='green', marker='X')
        plt.savefig("test_find_nearest_goal_pose.png")

    def test_find_nearest_goal_pose(self):
        return
        segments = np.array([
            [0, 0],
            [10, 0],
            [20, 0],
            [25, 5],
            [30, 10],
            [35, 18],
            [35, 25],
            [35, 35],
            [30, 41],
            [25, 43],
            [20, 43],
            [10, 43],
            [5, 33],
            [0, 23],
            [0, 13],
        ])

        path = []
        for p in segments:
            path.append(MapPose(x=p[0], y=p[1], z=0, heading=angle.new_deg(0)))

        points = np.array([
            [0, 0],
            [5, 2],
            [10, -1],
            [17, -3],
            [23, -4],
            [24, 6],
            [29, 8],
            [29, 17],
            [35, 18],
            [35, 20],
            [35, 22],
            [35, 24],
            [28, 34],
            [28, 39],
            [20, 39],
            [18, 39],
            [14, 39],
            [14, 29],
            [5, 24],
            [5, 20],
            [5, 19],
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
            11,  # 16
            13,
            14,
            14
        ], dtype=np.int32)

        self.plot_segments_and_points(segments, points, expected_pos)

        for i in range(len(expected_pos)):
            location = MapPose(x=points[i, 0], y=points[i, 1], z=0, heading=angle.new_deg(0))
            expected_pos_for_location = expected_pos[i]
            obtained_pos_for_location = MapPose.find_nearest_goal_pose(
                location, path, 0)

            self.assertEqual(expected_pos_for_location, obtained_pos_for_location,
                             f"wrong goal pose: expected: {expected_pos_for_location} ({points[expected_pos_for_location]}) obtained: {obtained_pos_for_location} ({points[obtained_pos_for_location]}) for test #{i}")

        

    def test_find_nearest_pose_bug(self):

        location = MapPose.from_str(
            "-89.151611328125|-3.723541498184204|0.029266204684972763|-15.790149688720703")
        path_str = ['-90.00924682617188|0.5093744993209839|0|0.2783680856227875', '-79.93721771240234|0.3|0|-0.518341064453125', '-69.8793716430664|0.3|0|-0.5555419921875', '-59.809959411621094|0.3|0|-0.555572509765625', '-49.68109130859375|0.3|0|-0.5505065321922302', '-39.657249450683594|0.3|0|-0.6009825468063354', '-29.611574172973633|0.3|0|-0.5670166015625', '-19.58110809326172|0.3|0|-0.5422667860984802',
                    '-9.545795440673828|0.4509453892707825|0|-0.5573425889015198', '0.5285676717758179|0.4524883061647415|0|-0.5581969618797302', '10.564472198486328|0.4549951829016209|0|-0.558319091796875', '20.591285705566406|-0.4232592135667801|0|-0.549591064453125', '30.688690185546875|-0.07684602588415146|0|0.14715316891670227', '40.688690185546875|-0.07684602588415146|0|0.14715316891670227', '50.688690185546875|-0.07684602588415146|0|0.14715316891670227']
        path = []
        for p in path_str:
            path.append(MapPose.from_str(p))

        pos = MapPose.find_nearest_goal_pose(
            location=location,
            poses=path,
            start=0
        )

        self.assertEqual(pos, 1)


if __name__ == "__main__":
    unittest.main()
