import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import angle
from pydriveless import MapPose

import cv2
from test_utils import fix_cv2_import
fix_cv2_import()



class TestMapNearestPose(unittest.TestCase):


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
            path.append(MapPose(x=p[0], y=p[1], z=0))

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
            [5, 24],    
            [5, 20],  
            [5, 19],
        ])

        expected_pos = np.array([
            1, 1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 7, 8, 9, 11, 11, 12, 13, 14, 14
        ], dtype=np.int32)
        
        for i in range (0, len(expected_pos)):
            location = MapPose(x=points[i, 0], y=points[i, 1], z=0)
            expected_pos_for_location = expected_pos[i]            
            obtained_pos_for_location = MapPose.find_nearest_goal_pose(location, path, min_distance=5, max_hopping=50)
            
            self.assertEqual(expected_pos_for_location, obtained_pos_for_location, 
                             f"wrong goal pose for test #{i} == location ({location.x}, {location.y}): expected: {expected_pos_for_location} ({segments[expected_pos_for_location]}) obtained: {obtained_pos_for_location} ({segments[obtained_pos_for_location]}) for test #{i}")


        self.plot_segments_and_points(segments, points, expected_pos)
        
        
    def test_find_nearest_pose(self):
        location = MapPose.from_str("-89.151611328125|-3.723541498184204|0.029266204684972763|-15.790149688720703")
        path_str = ['-90.00924682617188|0.5093744993209839|0|0.2783680856227875', '-79.93721771240234|0.3|0|-0.518341064453125', '-69.8793716430664|0.3|0|-0.5555419921875', '-59.809959411621094|0.3|0|-0.555572509765625', '-49.68109130859375|0.3|0|-0.5505065321922302', '-39.657249450683594|0.3|0|-0.6009825468063354', '-29.611574172973633|0.3|0|-0.5670166015625', '-19.58110809326172|0.3|0|-0.5422667860984802', '-9.545795440673828|0.4509453892707825|0|-0.5573425889015198', '0.5285676717758179|0.4524883061647415|0|-0.5581969618797302', '10.564472198486328|0.4549951829016209|0|-0.558319091796875', '20.591285705566406|-0.4232592135667801|0|-0.549591064453125', '30.688690185546875|-0.07684602588415146|0|0.14715316891670227', '40.688690185546875|-0.07684602588415146|0|0.14715316891670227', '50.688690185546875|-0.07684602588415146|0|0.14715316891670227']
        path = []
        for p in path_str:
            path.append(MapPose.from_str(p))
        
        res = MapPose.find_nearest_goal_pose(
            location=location,
            poses=path,
            start=0
        )
        
        self.assertEqual(res, 1)


    def test_find_nearest_pose_path_with_repeated_items(self):
        path_str = ["-90.0000991821289|-0.0008452492766082287|0.028222160413861275|0.00018598794633763798", "-89.45883330476649|-0.0008434922723435291|0.028222160413861275|0.00018598794633763798", "-88.91756742740407|-0.0008417352680788295|0.028222160413861275|0.00018598794633763798", "-88.3763019892927|0.13447649107679144|0.028222160413861275|0.00018598794633763798", "-87.83503655118135|0.2697947174216617|0.028222160413861275|0.00018598794633763798", "-87.29377111307|0.40511294376653195|0.028222160413861275|0.00018598794633763798", "-86.75250611420971|0.6757476394520078|0.028222160413861275|0.00018598794633763798", "-86.21124067609834|0.811065865796878|0.028222160413861275|0.00018598794633763798", "-85.66997523798699|0.9463840921417483|0.028222160413861275|0.00018598794633763798", "-85.26402626921625|1.0817018792355524|0.028222160413861275|0.00018598794633763798", "-84.72276083110488|1.2170201055804226|0.028222160413861275|0.00018598794633763798", "-84.18149539299353|1.352338331925293|0.028222160413861275|0.00018598794633763798", "-83.64022995488217|1.4876565582701633|0.028222160413861275|0.00018598794633763798", "-83.09896451677082|1.6229747846150333|0.028222160413861275|0.00018598794633763798", "-82.5576986394084|1.622976541619298|0.028222160413861275|0.00018598794633763798", "-82.01643276204597|1.6229782986235628|0.028222160413861275|0.00018598794633763798", "-81.47516644543248|1.487663586287222|0.028222160413861275|0.00018598794633763798", "-80.93390012881899|1.352348873950881|0.028222160413861275|0.00018598794633763798", "-80.52794984229504|1.0817172530228685|0.028222160413861275|0.00018598794633763798", "-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798", "-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798", "-79.98668352568156|0.9464025406865276|0.028222160413861275|0.00018598794633763798"]
        path = []
        for p in path_str:
            path.append(MapPose.from_str(p))

        location = MapPose(-79.92220306396484, 1.2667661905288696, 0.028616636991500854, angle.new_deg(-9.376293440256612))

        path_pos = MapPose.find_nearest_goal_pose(location, path, start=0, max_hopping=len(path) - 1)        
        self.assertEqual(path_pos, 20) # because of repeated items!

        path = MapPose.remove_repeated_seq_points_in_list(path)
        path_pos = MapPose.find_nearest_goal_pose(location, path, start=0, max_hopping=len(path) - 1)        
        self.assertEqual(path_pos, -1) # correcly found that we are after the path
        

if __name__ == "__main__":
    unittest.main()
