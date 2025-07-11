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

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

PLAN_TIMEOUT = 500
PLANNER_TYPE = LocalPlannerType.HybridAStar

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

def execute_plan (tst: CarlaTestUtils, ego: CarlaEgoCar, seq: int) -> None:
    coord = CoordinateConverter(COORD_ORIGIN)

    result = Telemetry.read_planning_result(seq)
   
    ds_path = downsample_waypoints(result.path)
    ideal_motion_path = coord.convert_waypoint_path_to_map_pose(result.ego_location, ds_path)
    
    tst.show_path(ideal_motion_path)
    
    controller = MotionController(
        period_ms=2,
        longitudinal_controller_period_ms=50,
        ego=ego,
        slam=CarlaSLAM(ego),
        on_finished_motion=on_finished_motion,
    )

    time.sleep(0.5)
    controller.set_path(ideal_motion_path, velocity=2.0)
    controller.start()


def main(argc: int, argv: List[str]) -> int:
    
    client = CarlaClient(town='Town07')
    tst = CarlaTestUtils(client)
    
    sb = ScenarioBuilder(client)
    path, ego = sb.load_scenario("../scenarios/cars_zigzag.sce", return_ego=True)
    ego.init_fake_bev_seg_camera()
    ego.set_brake(1.0)
    
    # ego = CarlaEgoCar(client)
    # ego.set_pose(-100, 0, 0, 0)
    # ego.set_brake(1.0)

    follower = ExpectatorCameraAutoFollow(client)
    follower.follow(ego.get_carla_ego_car_obj())
    execute_plan(tst, ego, 1)
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))