from model.slam import SLAM
from model.map_pose import VehiclePose
from .remote_car_control_client import RemoteSensorsClient
import math

class RemoteSLAM (SLAM):
    _sensors_client: RemoteSensorsClient

    def __init__(self, sensors_client: RemoteSensorsClient) -> None:
        super().__init__()
        self._sensors_client = sensors_client

    def estimate_ego_pose(self) -> VehiclePose:
        return self._sensors_client.get_pose()

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
            return RemoteSLAM.__DEG90
        elif dx == 0 and dy < 0:
            return -RemoteSLAM.__DEG90
        
        return 0.0

    def compute_heading(self) -> float:
        return self.__get_data().heading
    
    def compute_path_heading(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dy = p2.y - p1.y
        dx = p2.x - p1.x

        # maps in carla are inverted
        return RemoteSLAM.__eucl_compute_heading(dx, dy)
