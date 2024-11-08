import sys, os
sys.path.append("../../")
from carlasim.carla_client import CarlaClient
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.sensors.data_sensors import *
from scenario_builder import ScenarioBuilder
from planner.selfdrive_controller import SelfDriveController, PlanningDataBuilder, PlanningData, SelfDriveControllerResponse, SelfDriveControllerResponseType
#from slam.slam import SLAM
from carlasim.carla_slam import CarlaSLAM
from carlasim.expectator_cam_follower import ExpectatorCameraAutoFollow
from model.map_pose import MapPose
from model.world_pose import WorldPose
from planner.local_planner.local_planner import LocalPlannerType
from model.discrete_component import DiscreteComponent
from utils.telemetry import Telemetry

TEST_SPEED = 1.0

client = CarlaClient(town='Town07_Opt')


class AutoCameraSet (DiscreteComponent):
    _spectator: any
    _target: any
    _dist_m: float
    _client: CarlaClient
    _slam: CarlaSLAM
    
    def __init__(self, period_ms: int, client: CarlaClient) -> None:
        super().__init__(period_ms)
        self._client = client
        world = client.get_world()
        self._spectator = world.get_spectator()
        self._slam = None
        pass

    #

    def set_camera_addr(self, addr: tuple) -> None:
        x, y, z, pitch, yaw, roll = addr
        p = carla.Transform(carla.Location(x=x,y=y,z=z ),
                carla.Rotation( yaw = yaw, pitch = pitch, roll = roll))
        self._spectator.set_transform(p) 
    
    def __get_pos(self, i: int) -> tuple:
        match i:
            case 1:
               return (-75.676786422729492, 12.838396072387695, 31.128173828125, -116.02267456054688, 90, 180)
            case 2:
                return (-35.676786422729492, 12.838396072387695, 31.128173828125, -116.02267456054688, 90, 180)
            case 3:
                return (0.676786422729492, 12.838396072387695, 31.128173828125, -116.02267456054688, 90, 180)
            case 4:
                return (20.676786422729492, 12.838396072387695, 31.128173828125, -116.02267456054688, 90, 180)
        return None
    
    def pos1(self):
        addr = self.__get_pos(1)
        self.set_camera_addr(addr)

    def pos2(self):
        addr = self.__get_pos(2)
        self.set_camera_addr(addr)

    def pos3(self):
        addr = self.__get_pos(3)
        self.set_camera_addr( addr)

    def pos3(self):
        addr = self.__get_pos(4)
        self.set_camera_addr( addr)
        
    def auto_set(self, slam: CarlaSLAM):
        self.destroy()
        self._slam = slam
        self.start()
    
    def _loop(self, dt: float) -> None:
        if self._slam is None:
            return
        
        best = -1
        best_dist = 999999999
        for p in range(1, 5):
            l = self.__get_pos(p)
            location = self._slam.estimate_ego_pose()
            dist = MapPose.distance_between(MapPose(l[0], l[1], l[2], heading=0), location)
            if dist < best_dist:
                best_dist = dist
                best = p
        if best < 0:
            best = 1

        addr = self.__get_pos(best)
        self.set_camera_addr(addr)


def show_path(client: CarlaClient, path: list[MapPose]):
        world = client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=120000.0,
                                        persistent_lines=True)

def get_scenario_dir(scenario: int, planner: LocalPlannerType) -> str:
    planner_dir  = ""
    match planner:
        case LocalPlannerType.HierarchicalGroup:
            planner_dir = "h-ensemble"
        case LocalPlannerType.HybridAStar:
            planner_dir = "hybrid"
        case LocalPlannerType.Interpolator:
            planner_dir = "interpolator"
        case LocalPlannerType.Overtaker:
            planner_dir =  "overtaker"
        case LocalPlannerType.Ensemble:
            planner_dir = "p-ensemble"
        case LocalPlannerType.RRTStar:
            planner_dir = "rrt"
        
    return f"results/scen{scenario}/{planner_dir}"
        
        
        

def replay_scenario (client: CarlaClient, scenario: int, planner: LocalPlannerType):
    print(f"Replaying Scenario {scenario} planner {planner}")
    
    sb = ScenarioBuilder(client)
    path, ego = sb.load_scenario(f'scenarios/scenario{scenario}.sce', return_ego=True)
    ego.init_fake_bev_seg_camera()
    ego.set_brake(1.0)
    
    sdir = get_scenario_dir(scenario, planner)

    # Get a list of all files in the folder
    file_list = [f for f in os.listdir(sdir) if f.startswith('planning_result_') and f.endswith('.json')]
    # Sort the files by the number after 'planning_result_'
    file_list_sorted = sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    slam = CarlaSLAM(ego)
         
    slam.manual_calibrate(
            WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0))
    
    coord = slam.get_coordinate_converter()
    
    last = None
    for f in file_list_sorted:
        res = Telemetry.read_planning_result_from_file(f"{sdir}/{f}")
        rpath = coord.convert_waypoint_path_to_map_pose(res.ego_location, res.path)
        if len(rpath) > 0:
            last = rpath[-1]
        show_path(client, rpath)

    ego.set_pose(last.x, last.y, 2, last.heading+22)
            

    
    

replay_scenario(client=client, scenario=4, planner=LocalPlannerType.RRTStar)



