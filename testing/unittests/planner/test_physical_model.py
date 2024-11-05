import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest
from planner.physical_model import ModelCurveGenerator
from model.map_pose import MapPose
import matplotlib.pyplot as plt
import numpy as np
from model.waypoint import Waypoint
from testing.test_utils import TestFrame
from data.coordinate_converter import CoordinateConverter
from model.world_pose import WorldPose
import math

class TestPhysicalModel(unittest.TestCase):
    
    def test_path_generate(self):
        return
        model = ModelCurveGenerator()

        pose = MapPose(x=0.0, y=0.0, z=0.0, heading=0.0)
        path = model.gen_path_cg(pose, 10, 15, 20)
        self.plot_path("test_physical_output1.png", [path])
    
    def test_top_paths_generate(self):
        return
        model = ModelCurveGenerator()

        velocity = 10
        steps = 10
        pose = MapPose(x=0.0, y=0.0, z=0.0, heading=0.0)
        top_paths = model.gen_possible_top_paths(pose, velocity, steps)
        bottom_paths = model.gen_possible_bottom_paths(pose, velocity, steps)
        
        top_paths.extend(bottom_paths)
        self.plot_path("test_physical_output2.png", top_paths)  
    
    def plot_path(self, file: str, paths: list[list[MapPose]]):
        _, ax = plt.subplots()

        for path in paths:
            # Plot points
            x = np.zeros(len(path))
            y = np.zeros(len(path))
            
            i = 0
            for p in path:
                x[i] = p.x
                y[i] = p.y
                i += 1
                
            ax.scatter(x, y, color='red', marker='X')

        plt.savefig(file)
        
    def plot_waypoints(self, file: str, path: list[Waypoint], color = [255,255,255]) -> TestFrame:
        frame = TestFrame(256, 256)
        
        for p in path:
            frame.add_dot(p.x, p.z, color)
            
        frame.dump_to_file(file)
        return frame
        
    
    def test_discrete_path_generation_using_size(self):
        model = ModelCurveGenerator()
        l = MapPose(0, 0, 0, 0)
        w = WorldPose(0, 0, 0, 0)
        
        start = Waypoint(128, 128, heading=0)
        
        v = 1
        #num_steps = math.floor(120/v)
        path = model.gen_path_waypoint(start, v, 30, 100)
        # conv = CoordinateConverter(w)
        # path = conv.convert_map_path_to_waypoint(l, path)
        
        siz = Waypoint.distance_between(path[0], path[-1])
        print (f"path direct size: {siz} comming from 128,128 to {path[-1].x}, {path[-1].z}")
        print (f"path direct heading: {Waypoint.compute_heading(Waypoint(128,128), path[-1])}")
        frame = self.plot_waypoints("test_physical_waypoints_gen.png", path)
        
        end = path[-1]
        
        start.heading = 180
        path2 = model.connect_nodes_with_path(start, end, v)
        
        if path2 is not None:
            print(f"path connecting {start.x}, {start.z} and  {path[-1].x}, {path[-1].z} has {len(path2)} nodes")
            
            for p in path2:
                frame.add_dot(p.x, p.z, [0, 0, 255])        
                frame.dump_to_file("test_physical_waypoints_gen.png")
        
        
        
    def test_discrete_path_generation_using_size2(self):
        return
        model = ModelCurveGenerator()
        l = MapPose(0, 0, 0, 0)
        w = WorldPose(0, 0, 0, 0)
        
        path = model.gen_path_cg(l, 2, 30, 100)
        conv = CoordinateConverter(w)
        path = conv.convert_map_path_to_waypoint(l, path)
        
        dx = path[-1].x
        dz = path[-1].z
        siz = math.sqrt(dx ** 2 + dz ** 2)
        print (f"path direct size: {siz} comming from 128,128 to {path[-1].x}, {path[-1].z}")
        print (f"path direct heading: {Waypoint.compute_heading(Waypoint(128,128), path[-1])}")
        
        
        

        self.plot_waypoints("test_physical_waypoints_gen.png", path)

if __name__ == "__main__":
    unittest.main()
