import sys

sys.path.append("../../../")
from typing import List
from model.map_pose import MapPose
from model.world_pose import WorldPose
from slam.slam import SLAM
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType
from model.planning_data import PlanningData, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from utils.telemetry import Telemetry
from motion.motion_controller import MotionController
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.carla_slam import CarlaSLAM
from carlasim.sensors.data_sensors import *
from model.waypoint import Waypoint
from testing.unittests.carla_test_utils import CarlaTestUtils
from carlasim.expectator_cam_follower import ExpectatorCameraAutoFollow
import time
from scenario_builder import ScenarioBuilder, ScenarioActor
from planner.collision_detector import CollisionDetector
from planner.planning_data_builder import PlanningDataBuilder, PlanningData

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = 500
PLANNER_TYPE = LocalPlannerType.HybridAStar

class DataBuilder(PlanningDataBuilder):
    _seq: int
    
    def set_seq(self, seq: int) -> None:
        self._seq = seq
    
    def build_planning_data(self) -> PlanningData:
        bev = Telemetry.read_collision_bev(self._seq)
        report = Telemetry.read_collision_report(1)
        
        return PlanningData(
            bev=bev,
            ego_location=report.ego_location,
            velocity=5.0,
            goal=report.watch_target,
            next_goal=None
        )
        
        
        
        
class TestSLAM(SLAM):    
    _pose: MapPose
    
    def __init__(self):
        super().__init__(None, None, None)
    
    def set_pose(self, pose: MapPose) -> None:
        self._pose = pose
    
    def estimate_ego_pose (self) -> MapPose:
        return self._pose

def on_finished_motion(controller: MotionController):
    print ("finished motion")
    controller.brake()
    pass

def downsample_waypoints(waypoints: List[Waypoint]) -> List[Waypoint]:
    res = []
    division = max(1, math.floor(len(waypoints) / 20))

    i = 0
    for p in waypoints:
        if i % division == 0:
            res.append(p)
            i += 1

    if len(waypoints) > 0:
        res.append(waypoints[len(waypoints) - 1])
    return res

def on_collision_detected() -> None:
    print ("Collision detected")

#def watch_path (tst: CarlaTestUtils, ego: CarlaEgoCar, seq: int) -> None:
def watch_path (seq: int) -> None:    
    coord = CoordinateConverter(COORD_ORIGIN)

    result = Telemetry.read_planning_result(seq)
   
    ds_path = downsample_waypoints(result.path)
    ideal_motion_path = coord.convert_waypoint_path_to_map_pose(result.ego_location, ds_path)
    
    #tst.show_path(ideal_motion_path)
    
    data_builder = DataBuilder()
    data_builder.set_seq(1)
    
    slam = TestSLAM()
    slam.set_pose(result.ego_location)
    
    cd = CollisionDetector(
        period_ms=150,
        coordinate_converter=coord,
        planning_data_builder=data_builder,
        slam = slam,
        on_collision_detected_cb=on_collision_detected,
        with_telemetry=False
    )
  
  
    cd.watch_path(ideal_motion_path)
    cd.start()
    
    print("press enter to destroy...")
    input()
    cd.destroy()
    


def main(argc: int, argv: List[str]) -> int:
    
    
    
    # client = CarlaClient(town='Town07')
    # tst = CarlaTestUtils(client)
    
    # sb = ScenarioBuilder(client)
    # path, ego = sb.load_scenario("../scenarios/cars_zigzag.sce", return_ego=True)
    # ego.init_fake_bev_seg_camera()
    # ego.set_brake(1.0)
    
    # # ego = CarlaEgoCar(client)
    # # ego.set_pose(-100, 0, 0, 0)
    # # ego.set_brake(1.0)

    # follower = ExpectatorCameraAutoFollow(client)
    # follower.follow(ego.get_carla_ego_car_obj())
    watch_path(1)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))