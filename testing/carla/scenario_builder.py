import sys
import os
sys.path.append("../../")
from carlasim.carla_object_summon import CarlaObjectSummoner
from carlasim.carla_slam import CarlaSLAM
from carlasim.carla_ego_car import CarlaEgoCar
from carlasim.carla_client import CarlaClient
from model.map_pose import MapPose
from model.waypoint import Waypoint
from data.coordinate_converter import CoordinateConverter
import cv2, time, carla
from vision.occupancy_grid_cuda import OccupancyGrid
from motion.motion_controller import MotionController

class PathBuilder:
    client: CarlaClient
    ego: CarlaEgoCar
    coord_conv: CoordinateConverter
    slam: CarlaSLAM
    motion: MotionController
    _in_motion: bool
    
    OPERATIONAL_FRAME_FILE = "frame.png"
    REAL_WIDTH = 34.641016151377535
    REAL_HEIGHT = 34.641016151377535
    OG_WIDTH = 256
    OG_HEIGHT = 256

    def __init__(self, client: CarlaClient, color: str = '120, 0, 255'):
        self.client = client
        self.ego = CarlaEgoCar(self.client, color)
        self.ego.set_power(0)
        self.ego.set_steering(0)
        self.ego.set_brake(1.0)
        self.ego.init_fake_bev_seg_camera(PathBuilder.OG_WIDTH, PathBuilder.OG_HEIGHT)
        
        self.coord_conv = CoordinateConverter(
            PathBuilder.REAL_WIDTH, 
            PathBuilder.REAL_HEIGHT, 
            PathBuilder.OG_WIDTH, 
            PathBuilder.OG_HEIGHT
        )
        
        self.slam = CarlaSLAM(self.ego)
        self.motion =MotionController(period_ms=1, 
                                      longitudinal_controller_period_ms=50, 
                                      on_invalid_path=self.__invalid_path,
                                      on_finished_motion=self.__finished_motion,
                                      odometer=lambda : self.ego.get_odometer().read(),
                                      power_actuator=lambda p: self.ego.set_power(p),
                                      brake_actuator=lambda p: self.ego.set_brake(p),
                                      slam=self.slam,
                                      steering_actuator=lambda p: self.ego.set_steering(p)
                                      )
        self._in_motion = False
        self.motion.start()
        self.motion.brake()
        
    def __invalid_path(self):
        self._in_motion = False
        print ("INVALID PATH!")
        
    def __finished_motion(self):
        print ("finished motion")
        self.motion.cancel()
        self.motion.brake()
        self._in_motion = False

    def set_pose_to_last_path_pos(self, pos: int):
        log_file = f"log/path_{pos}.log"
        if not os.path.exists(log_file):
            print (f"log not found for pos = {pos} at {log_file}")
            return
        
        path = self.__decode_path_from_log(log_file)
        p = path[-1]
        self.ego.set_pose(p.x, p.y, 2, p.heading)
        time.sleep(2)
        
        l = self.ego.get_location()
        x = round(l[0], 2)
        y = round(l[1], 2)
        z = round(l[2], 2)
        h = round(self.ego.get_heading(), 2)
        
        print (f"car pos: {x}, {y}, {z}, heading: {h}")


    def set_pose(self, pose: list[float]):
        self.ego.set_pose(pose[0], pose[1], pose[2], pose[3])
        time.sleep(2)

    def extract_frame(self):
        self.__capture_bev_to(PathBuilder.OPERATIONAL_FRAME_FILE)
    
    def __capture_bev_to(self, file: str):
        f = self.ego.get_bev_camera().read()
        f = OccupancyGrid(f).get_color_frame()
        cv2.imwrite(file, f)
    
    def __read_path (self) -> list[MapPose]:
        if not os.path.exists(PathBuilder.OPERATIONAL_FRAME_FILE):
            return []
        
        frame = cv2.imread(PathBuilder.OPERATIONAL_FRAME_FILE)
        
        path = []
        for i in range (0, frame.shape[0]):
            for j in range (0, frame.shape[1]):
                if frame[i, j, 0] == 255 and \
                    frame[i, j, 1] == 255 and \
                    frame[i, j, 2] == 255:
                    waypoint = Waypoint(j, i)
                    path.append(waypoint)
        
        path.reverse()
    
        return self.coord_conv.convert_path_to_world_pose(self.slam.estimate_ego_pose(), path, 10.0)
    
    def __find_next_seq(self) -> int:
        if not os.path.exists("log/"):
            os.system("mkdir log")
            return 1
        else:
            files = os.listdir("log/")
            max_pos = 0
            for f in files:
                if "png" in f:
                    if "orig" in f: continue
                    p = int(f.replace(".png", ""))
                    max_pos = max(max_pos, p)
            return max_pos
        
    def __save_path_to_log(self, pos: int, path: list[MapPose]):
        log = open(f"log/path_{pos}.log", "w")
        for p in path:
            log.write(f"{str(p)}\n")
        log.close()
    
    def save_path(self, pos: int) -> int:
        # pos = self.__find_next_seq()
        print (f"saving path on position: {pos}")
        os.system(f"cp frame.png log/{pos}.png")
        self.__save_path_to_log(pos, self.__read_path())
        self.__capture_bev_to(f"log/orig_{pos}.png")
        return pos
    
    def __decode_path_from_log(self, log_file: str) -> list[MapPose]:
        path = []
        log = open(log_file, "r")
        
        lines = log.readlines()
        for l in lines:
            if l is None or l == "": continue
            path.append(MapPose.decode(l))
        
        return path
    
    def show_path_on_simulator(self, log_file: str):
        if not os.path.exists(log_file):
            print (f"log not found at {log_file}")
            return
        
        path = self.__decode_path_from_log(log_file)
        
        world = self.client.get_world()
        for p in path:
            world.debug.draw_string(carla.Location(p.x, p.y, 2), 'X', draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=60.0,
                                        persistent_lines=True)
    
    def move_on_path(self, log_file: str) -> None:
        if not os.path.exists(log_file):
            print (f"log not found at {log_file}")
            return
        
        path = self.__decode_path_from_log(log_file)
        
        
        self.set_pose([path[0].x, path[1].y, 2, path[2].heading])
        time.sleep(2)
        
        location = self.ego.get_location()
        heading = self.ego.get_heading()
        
        self.motion.set_path(path)
        self._in_motion = True
        
        while self._in_motion:
            time.sleep(0.1)
        
        time.sleep(1)
        self.ego.set_pose(location[0], location[1], location[2], heading)
    
    
class ScenarioActor:
    type: str
    x: float
    y: float
    z: float
    heading: float
    color: str
    
    def __init__(self, type: str, x: float, y: float, z: float, heading: float, color: str):
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.color = color
    
    
    def decode(payload: str) -> 'ScenarioActor':
        p = payload.split("|")
        return ScenarioActor(
            str(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3]),
            float(p[4]),
            str(p[5])
        )

    def __str__(self) -> str:
        return f"{self.type}|{self.x}|{self.y}|{self.z}|{self.heading}|{self.color}"
      
class ScenarioBuilder:
    
    TYPE_EGO = "ego"
    TYPE_CAR = "car"
    TYPE_CONE = "cone"
    TYPE_GOAL = "goal"
    
    _client: CarlaClient
    _summoner: CarlaObjectSummoner
    _actors: list[ScenarioActor]
     
    def __init__(self, client):
        self._client = client
        self._summoner = CarlaObjectSummoner(client)
        self._actors = []

    def add_car(self, pose: list[float], color: str = '144, 238, 144') -> None:
        self._actors.append(ScenarioActor(ScenarioBuilder.TYPE_CAR, pose[0], pose[1], pose[2], pose[3], color))
        self._summoner.add_car(pose[0], pose[1], pose[2], pose[3], color)

    def add_cone(self, pose: list[float]) -> None:
        self._actors.append(ScenarioActor(ScenarioBuilder.TYPE_CONE, pose[0], pose[1], pose[2], 0, "-"))
        self._summoner.add_cone(pose[0], pose[1], pose[2])
    
    def add_ego(self, pose: list[float], color: str = '0, 255, 0') -> None:
        self._actors.append(ScenarioActor(ScenarioBuilder.TYPE_EGO, x=pose[0], y=pose[1], z=pose[2], heading=pose[3], color=color))
        self._summoner.add_car(self._ego_start.x, self._ego_start.y, self._ego_start.z, self._ego_start.heading, self._ego_start.color)

    def clear(self) -> None:
        self._actors.clear()
        self._summoner.clear_objects()
        
    def add_goal(self, pos_x_y: list[float]) -> None:
        p = ScenarioActor(ScenarioBuilder.TYPE_GOAL, x=pos_x_y[0], y=pos_x_y[1], z=2, heading=0, color="")
        self._actors.append(p)
        self.__show_goal(p)

    def __show_goal(self, actor: ScenarioActor, i: int) -> any:
        world = self._client.get_world()
        world.debug.draw_string(carla.Location(actor.x, actor.y, actor.z), f'{i}', draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=30.0,
                                        persistent_lines=True)

        
    def save_scenario(self, file: str) -> None:
        log = open(file, "w")
        
        for p in self._actors:
            log.write(f"{str(p)}\n")
            
        log.close()
    
    @classmethod
    def __decode_scenario_from_log(cls, log_file: str) -> list[ScenarioActor]:
        path = []
        log = open(log_file, "r")
        
        lines = log.readlines()
        for l in lines:
            l = l.replace("\n", "")
            if l is None or l == "": continue
            path.append(ScenarioActor.decode(l))
        
        return path
    
    @classmethod
    def read_goal_list(cls, log_file: str) -> list[MapPose]:
        if not os.path.exists(log_file):
            print (f"[load_scenario failed] log not found at {log_file}")
            return
        
        actors = ScenarioBuilder.__decode_scenario_from_log(log_file)
        
        path = []
        
        for actor in actors:
            if actor.type == ScenarioBuilder.TYPE_GOAL:
                path.append(MapPose(
                    x = actor.x,
                    y = actor.y,
                    z = actor.z,
                    heading = actor.heading
                ))
        return path
        
    
    def load_scenario(self, log_file: str, return_ego: bool = False):
        self.clear()
        if not os.path.exists(log_file):
            print (f"[load_scenario failed] log not found at {log_file}")
            return
        
        self._actors = ScenarioBuilder.__decode_scenario_from_log(log_file)
        
        ego: CarlaEgoCar = None
        path: list[MapPose] = []
        
        i = 0
        for actor in self._actors:
            i += 1
            if actor.type == ScenarioBuilder.TYPE_EGO:
                if return_ego:
                    ego = CarlaEgoCar(self._client)
                    ego.init_fake_bev_seg_camera()
                    ego.set_pose(actor.x, actor.y, actor.z, actor.heading)
                else:                   
                    self._summoner.add_car(actor.x, actor.y, actor.z, actor.heading)
            elif actor.type == ScenarioBuilder.TYPE_CAR:
                self._summoner.add_car(actor.x, actor.y, actor.z, actor.heading, actor.color)
            elif actor.type == ScenarioBuilder.TYPE_CONE:
                self._summoner.add_cone(actor.x, actor.y, actor.z)
            elif actor.type == ScenarioBuilder.TYPE_GOAL:
                self.__show_goal(actor, i)
                path.append(MapPose(
                    x = actor.x,
                    y = actor.y,
                    z = 0,
                    heading = actor.heading
                ))

        return path, ego