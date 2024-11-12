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



class AutoCameraSet (DiscreteComponent):
    _spectator: any
    _target: any
    _dist_m: float
    _client: CarlaClient
    _slam: CarlaSLAM
    _last_pos: int
    
    def __init__(self, period_ms: int, client: CarlaClient) -> None:
        super().__init__(period_ms)
        self._client = client
        world = client.get_world()
        self._spectator = world.get_spectator()
        self._slam = None
        self._camera_pos = np.zeros(6)
        self._last_pos = -1
        pass

    #

    def set_camera_addr(self, addr: tuple) -> None:
        x, y, z, pitch, yaw, roll = addr
        p = carla.Transform(carla.Location(x=x,y=y,z=z ),
                carla.Rotation( yaw = yaw, pitch = pitch, roll = roll))
        self._spectator.set_transform(p) 

    def set_full_path_cam(self):
        addr = [-2.7538657188415527, -114.45896911621094, 131.358322143554, -86.02267456054688, 0, 175]
        self.set_camera_addr(addr)

    
    def __get_pos(self, i: int) -> list:
        if i >= 1 and i <= 5:
            base_camera = [-75.676786422729492, 12.838396072387695, 31.128173828125, -116.02267456054688, 90, 180]
            base_camera[0] += (i-1) * 35
            return base_camera
        if i == 6:
            return [54.5118408203125, -37.8912239074707, 43.70728302001953, -126.02267456054688, 190, 180]
        if i == 7:
            return [52.82429122924805, -74.19267272949219, 36.18389892578125, -136.02267456054688, -110, 180]
        if i == 8:
            return [52.82429122924805, -74.59267272949219, 36.18389892578125, -136.02267456054688, 90, 180]
        if i == 9:
            return [49.8397216796875, -166.0282440185547, 64.50352096557617, -110.02267456054688, -40, 180]
        if i == 10:
            return [34.9863395690918, -201.5182342529297, 62.7861328125, -100.02267456054688, -100, 180]
        if i == 11:
            return [54.60447692871094, -239.78192138671875, 56.99696350097656, -120.02267456054688, -40, 180]
        return None
    
           
    def auto_set(self, slam: CarlaSLAM):
        self.destroy()
        self._slam = slam
        self.start()


    def _loop(self, dt: float) -> None:
        if self._slam is None:
            return
        
        best = -1
        best_dist = 999999999
        for p in range(1, 12):
            l = self.__get_pos(p)
            if l is None:
                continue
            location = self._slam.estimate_ego_pose()
            dist = MapPose.distance_between(MapPose(l[0], l[1], l[2], heading=0), location)
            if dist < best_dist:
                best_dist = dist
                best = p
        if best < 0:
            best = 1
            
        if self._last_pos != best:
            print (f"********************")
            print (f"changed pos to: {best}")
            print (f"********************\n\n")
        self._last_pos = best

        addr = self.__get_pos(best)
        self.set_camera_addr(addr)

client = CarlaClient(town='Town07_Opt')
auto_camera = AutoCameraSet(100, client)


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
        unseg_bev = self._ego.get_rgb_bev_camera().read()
        location_useg = self._slam.estimate_ego_pose()

        return PlanningData(
            unseg_bev=unseg_bev,
            bev=bev,
            ego_location=location,
            velocity=TEST_SPEED,
            goal=None,
            next_goal=None,
            ego_location_ubev=location_useg
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
            time.sleep(4)
            auto_camera.destroy()
            # auto_camera.set_full_path_cam()
            return  
        case SelfDriveControllerResponseType.VALID_WILL_EXECUTE:
            show_path(client, res.motion_path)
            return

def drive_scenario (client: CarlaClient, file: str):
    print(f"Loading Scenario {file}")
    
    
    
    sb = ScenarioBuilder(client)
    path, ego = sb.load_scenario(file, return_ego=True)
    ego.init_dual_bev_camera()
    ego.set_brake(1.0)

    
    follower = None
    # follower = ExpectatorCameraAutoFollow(client)
    # follower.follow(ego.get_carla_ego_car_obj())
    
    print(f"Self-driving EGO vehicle through a global path with #{len(path)} goals")
    
    data_builder = CarlaPlanningDataBuilder(ego)
    auto_camera.auto_set(data_builder.get_slam())
    
    #ego.set_pose(67.14884185791016,-116.3796615600586,8.809161186218262,-86.6710205078125)
    
    controller = SelfDriveController(
        ego=ego,
        planning_data_builder=data_builder,
        controller_response=controller_response,
        slam=data_builder.get_slam(),
        local_planner_type=LocalPlannerType.HierarchicalGroup
    )
    
    time.sleep(2)
    
    controller.start()
    controller.drive(path)
    return controller, follower, ego


# BUG NO COLLISION DETECTOR!
#controller, follower, ego = drive_scenario(client=client, file="scenarios/turn_right_obstacle.sce")

os.system("rm /home/cristiano/Documents/Projects/Mestrado/code/selfdrive/testing/carla/planning_data/* ")

controller, follower, ego = drive_scenario(client=client, file="scenarios/scenario5.sce")


print ("press enter to destroy")
input()

controller.destroy()
#follower.destroy()
ego.destroy()

print ("the simulation has ended")

