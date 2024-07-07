from model.slam import SLAM
from model.map_pose import VehiclePose
from carlasim.carla_ego_car import EgoCar
import math

class CarlaSLAM (SLAM):
    __DEG90 = math.pi / 2
    _car: EgoCar

    def __init__(self, car: EgoCar) -> None:
        self._car = car
        pass

    def estimate_ego_pose(self) -> VehiclePose:
        location = self._car.get_location()
        heading = self._car.get_heading()
        pose = VehiclePose(location[0], location[1], heading, 0)
        pose.z = location[2]
        return pose

    def distance_to(self, goal: VehiclePose) -> float:
        return self.estimate_ego_pose().distance_to(goal)
        
    #
    #     Q4   |   Q3
    #      ----+-------> +x
    #     Q2   |   Q1
    #          |
    #          +y

    def __eucl_compute_heading(dx: float, dy: float) -> float:
        if dy >= 0 and dx > 0:                      # Q1
            return math.atan(dy/dx)
        elif dy >= 0 and dx < 0:                    # Q2
            return math.pi - math.atan(dy/abs(dx))
        elif dy < 0 and dx > 0:                     # Q3
            return  -math.atan(abs(dy)/dx)
        elif dy < 0 and dx < 0:                     # Q4
            return math.atan(dy/dx) - math.pi
        elif dx == 0 and dy > 0:
            return CarlaSLAM.__DEG90
        elif dx == 0 and dy < 0:
            return -CarlaSLAM.__DEG90
        
        return 0.0

    def compute_heading(self) -> float:
        return  self._car.get_heading()
    
    def compute_path_heading(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dy = p2.y - p1.y
        dx = p2.x - p1.x

        # maps in carla are inverted
        return CarlaSLAM.__eucl_compute_heading(dx, dy)