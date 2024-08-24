import sys, os
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
from scenario_builder import ScenarioBuilder, ScenarioActor
from data_logger import DataLogger
from vision.occupancy_grid_cuda import OccupancyGrid
from model.waypoint import Waypoint
from model.map_pose import MapPose
import cv2, time
from planner.selfdrive_controller import SelfDriveController, PlanningDataBuilder, PlanningData, SelfDriveControllerResponse, SelfDriveControllerResponseType
from slam.slam import SLAM

class CarlaPlanningDataBuilder(PlanningDataBuilder):
    
    _ego: CarlaEgoCar
    _slam: SLAM
    
    def __init__(self, ego: CarlaEgoCar) -> None:
        super().__init__()
        self._ego = ego

        self._slam =SLAM(gps=ego.get_gps(), 
                          imu=ego.get_imu(), 
                          odometer=ego.get_odometer())       
         
        self._slam.calibrate()
    
    def build_planning_data(self) -> PlanningData:
        return PlanningData(
            bev=self._ego.get_bev_camera().read(),
            ego_location=self._slam.estimate_ego_pose(),
            velocity=10.0,
            goal=None,
            next_goal=None
        )
        
    def get_slam(self) -> SLAM:
        return self._slam

def controller_response(self, res: SelfDriveControllerResponse) -> None:
    pass

def drive_scenario (client: CarlaClient, file: str):
    print(f"Loading Scenario {file}")
    
    sb = ScenarioBuilder(client)
    path, ego = sb.load_scenario(file, return_ego=True)
    ego.init_fake_bev_camera()
    ego.set_brake(1.0)
    
    print(f"Self-driving EGO vehicle through a global path with #{len(path)} goals")
    
    data_builder = CarlaPlanningDataBuilder(ego)
    
    controller = SelfDriveController(
        ego=ego,
        planning_data_builder=data_builder,
        controller_response=controller_response,
        slam=data_builder.get_slam()
    )
    
    controller.start()
    controller.drive(path)


client = CarlaClient(town='Town07')
drive_scenario(client=client, file="scenarios/cars_zigzag.sce")