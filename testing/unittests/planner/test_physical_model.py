import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest
from planner.physical_model import ModelCurveGenerator
from model.map_pose import MapPose
import matplotlib.pyplot as plt
import numpy as np

class TestPhysicalModel(unittest.TestCase):
    
    def test_path_generate(self):
        model = ModelCurveGenerator()

        pose = MapPose(x=0.0, y=0.0, z=0.0, heading=0.0)
        path = model.gen_path_cg(pose, 10, 15, 20)
        self.plot_path("test_physical_output1.png", [path])
    
    def test_top_paths_generate(self):
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
    

if __name__ == "__main__":
    unittest.main()
