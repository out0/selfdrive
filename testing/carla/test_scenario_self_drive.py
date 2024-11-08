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

    def auto_set(self, slam: CarlaSLAM):
        self.destroy()
        self._slam = slam
        self.start()
    
    def _loop(self, dt: float) -> None:
        if self._slam is None:
            return
        
        best = -1
        best_dist = 999999999
        for p in range(1, 4):
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


class CarlaPlanningDataBuilder(PlanningDataBuilder):
    
    _ego: CarlaEgoCar
    _slam: CarlaSLAM
    
    def __init__(self, ego: CarlaEgoCar) -> None:
        super().__init__()
        self._ego = ego

        # self._slam =SLAM(gps=ego.get_gps(), 
        #                   imu=ego.get_imu(), 
        #                   odometer=ego.get_odometer())       
        
        self._slam = CarlaSLAM(ego)
         
        self._slam.manual_calibrate(
            WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0))
    
    def build_planning_data(self) -> PlanningData:
        location = self._slam.estimate_ego_pose()
        bev = self._ego.get_bev_camera().read()
        return PlanningData(
            bev=bev,
            ego_location=location,
            velocity=TEST_SPEED,
            goal=None,
            next_goal=None
        )
        
    def get_slam(self) -> CarlaSLAM:
        return self._slam

def show_path(client: CarlaClient, path: list[MapPose]):
        world = client.get_world()
        for w in path:
            world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=120000.0,
                                        persistent_lines=True)

def controller_response(res: SelfDriveControllerResponse) -> None:
    match res.response_type:
        case SelfDriveControllerResponseType.UNKNOWN_ERROR:
            print("[Vehicle Controller callback] unknown error")
            return
        case SelfDriveControllerResponseType.CANT_LOCATE_IN_GLOBAL_PATH:
            print("[Vehicle Controller callback] cant locate ego car in global path")
            return
        case SelfDriveControllerResponseType.PLAN_RETURNED_NONE:
            print("[Vehicle Controller callback] The local planner returned NONE as result. Bug?")
            return            
        case SelfDriveControllerResponseType.PLAN_INVALID_START:
            print("[Vehicle Controller callback] The local planner got invalid start. The car is stuck.")
            return
        case SelfDriveControllerResponseType.PLAN_INVALID_GOAL:
            print("[Vehicle Controller callback]  The local planner got an invalid goal. A global replan may solve the problem.")
            return            
        case SelfDriveControllerResponseType.PLAN_INVALID_PATH:
            print("[Vehicle Controller callback] The local planner got an invalid path. No local planner was able to solve for this OG")
            return               
        case SelfDriveControllerResponseType.MOTION_INVALID_PATH:
            print("[Vehicle Controller callback] The motion planner got an invalid path. Drift? bug?")
            return              
        case SelfDriveControllerResponseType.GOAL_REACHED:
            print("[Vehicle Controller callback] The final goal was reached successfuly \o/")
            return  
        case SelfDriveControllerResponseType.VALID_WILL_EXECUTE:
            show_path(client, res.motion_path)
            return

def drive_scenario (client: CarlaClient, file: str):
    print(f"Loading Scenario {file}")
    
    
    
    sb = ScenarioBuilder(client)
    path, ego = sb.load_scenario(file, return_ego=True)
    ego.init_fake_bev_seg_camera()
    ego.set_brake(1.0)
    

    
    follower = None
    # follower = ExpectatorCameraAutoFollow(client)
    # follower.follow(ego.get_carla_ego_car_obj())
    
    print(f"Self-driving EGO vehicle through a global path with #{len(path)} goals")
    
    data_builder = CarlaPlanningDataBuilder(ego)
    
    auto_camera = AutoCameraSet(100, client)
    auto_camera.auto_set(data_builder.get_slam())
    
    controller = SelfDriveController(
        ego=ego,
        planning_data_builder=data_builder,
        controller_response=controller_response,
        slam=data_builder.get_slam(),
        local_planner_type=LocalPlannerType.RRTStar
    )
    
    controller.start()
    controller.drive(path)
    return controller, follower, ego


# BUG NO COLLISION DETECTOR!
#controller, follower, ego = drive_scenario(client=client, file="scenarios/turn_right_obstacle.sce")

os.system("rm /home/cristiano/Documents/Projects/Mestrado/code/selfdrive/testing/carla/planning_data/* ")

controller, follower, ego = drive_scenario(client=client, file="scenarios/scenario3.sce")


print ("press enter to destroy")
input()

controller.destroy()
#follower.destroy()
ego.destroy()

print ("the simulation has ended")

