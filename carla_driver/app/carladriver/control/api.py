import carla
import random
from threading import Thread
from pydriveless import MapPose
import time
from .session import CarlaSession
from .ego import CarlaEgoVehicle



class CarlaVirtualPath:
    __path: list[MapPose]
    __path_show_thr: Thread
    __run: bool
    
    def __init__(self, session: CarlaSession, path: list[MapPose], color: tuple[int, int, int] = [255, 0, 0]):
        self.__session = session
        self.__path = path
        self.__color = color
        self.__path_show_thr = Thread(target=self.__show_path, daemon=True)
        self.__path_show_thr.start()

    def __del__(self):
        self.__run = False
        if self.__path is None:
            return

        if self.__path_show_thr is None or not self.__path_show_thr.is_alive():
            self.__path = None
            return

        self.__path_show_thr.join()
        self.__path.clear()
        self.__path = None
    
    def __show_path(self) -> None:
        self.__run = True
        while self.__run:
            for p in self.__path:
                if not self.__run:
                    break
                x = p.x
                y = p.y
                z = p.z
                symbol = 'O'
                self.__session.world.debug.draw_string(carla.Location(x, y, z), symbol, draw_shadow=False,
                                    color=carla.Color(r=self.__color[0], g=self.__color[1], b=self.__color[2]), life_time=1.0,
                                    persistent_lines=True)
            time.sleep(1)


class CarlaVirtualPose:
    __pose: MapPose
    __point_show_thr: Thread
    __run: bool
    
    def __init__(self, session: CarlaSession, pose: MapPose, color: tuple[int, int, int] = [0, 255, 0]):
        self.__session = session
        self.__pose = pose
        self.__color = color
        self.__point_show_thr = Thread(target=self.__show_point, daemon=True)
        self.__point_show_thr.start()

    def __del__(self):
        self.__run = False
        self.__point_show_thr.join()
        self.__pose = None
    
    def __show_point(self) -> None:
        self.__run = True
        while self.__run:
            x = self.__pose.x
            y = self.__pose.y
            z = self.__pose.z
            symbol = 'X'
            self.__session.world.debug.draw_string(carla.Location(x, y, z), symbol, draw_shadow=False,
                        color=carla.Color(r=self.__color[0], g=self.__color[1], b=self.__color[2]), life_time=1.0,
                        persistent_lines=True)
            time.sleep(1)
 


class CarlaSimulation:
    _session: any
    _objects: dict
    _paths: list[CarlaVirtualPath]
    _poses: list[CarlaVirtualPose]
    
    def __init__(self, town_name: str, ip: str = 'localhost', port: int = 2000, timeout: float = 110.0):
        self._objects = {}
        self._paths = []
        self._poses = []
        self._session = CarlaSession()
        self._session.client = carla.Client(ip, port)
        self._session.client.set_timeout(timeout)
        self._session.world = self._session.client.get_world()
        
        try:
            current_map_name = self._session.world.get_map().name
        except:
            self._session.client.load_world(town_name)
            self._session.world = self._session.client.get_world()
            current_map_name = self._session.world.get_map().name

            
        if town_name not in current_map_name:
            self._session.client.load_world(town_name)
            self._session.world = self._session.client.get_world()
        else:
            print(f"Current map is already {current_map_name}. No need to load {town_name}.")
            
        self._session.blueprint_lib = self._session.world.get_blueprint_library()

        pass
       
    def add_object(self, name: str, type: str, pos: tuple[float, float, float], rotation: tuple[float, float, float] = [0, 0, 0], color = None) -> None:
        self.remove_object(name)
        bp =  self._session.blueprint_lib.find(type)

        if color is None:
            color = tuple(random.randint(0, 255) for _ in range(3))
        bp.set_attribute('color', f"{color[0]}, {color[1]}, {color[2]}")

        x, y , z = pos
        pitch, yaw, roll = rotation 
        t = carla.libcarla.Transform( carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(pitch, yaw, roll))
        obj = self._session.world.spawn_actor(bp, t)
        self._objects[name] = obj
        
    def add_ego_vehicle(self, pos: tuple[float, float, float], rotation: tuple[float, float, float] = [0, 0, 0], vehicle_type: str = 'vehicle.tesla.model3', color = None) -> CarlaEgoVehicle:        
        self.add_object('ego_vehicle', vehicle_type, pos, rotation, color)
        vehicle = self.get_object('ego_vehicle')
        if vehicle is None:
            raise ValueError("Ego vehicle could not be created. Check the type and parameters.")        
        return CarlaEgoVehicle(self._session, vehicle)
    
    def get_world(self) -> carla.World:
        return self._session.world
    
    def get_ego_vehicle(self) -> CarlaEgoVehicle:
        vehicle = self.get_object('ego_vehicle')
        if vehicle is None:
            raise ValueError("Ego vehicle does not exist. Create it first.")
        return CarlaEgoVehicle(self._session, vehicle)

    def remove_object(self, name: str) -> None:
        if name in self._objects:
            obj = self._objects[name]
            obj.destroy()
            self._objects[name] = None
            
    def clear_objects(self) -> None:
        for o in self._objects.values():
            o.destroy()
        self._objects.clear()
        
    def move_spectator (self, pos: tuple[float, float, float], rotation: tuple[float, float, float] = [0, 0, 0]) -> None:
        x, y , z = pos
        pitch, yaw, roll = rotation 
        t = carla.libcarla.Transform( carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(pitch, yaw, roll))
        self._session.world.get_spectator().set_transform(t)
        
    
    def get_object(self, name: str):
        if name in self._objects:
            return self._objects[name]
        return None
    
    def show_coordinate(self, pose: tuple[float, float, float],  color = [0, 255, 0]) -> None:
        self._poses.append(CarlaVirtualPose(self._session,  MapPose(pose[0], pose[1], pose[2]), color))
        

    def show_path(self, path: list[MapPose], color = [255, 0, 0]) -> None:
        self._paths.append(CarlaVirtualPath(self._session, path, color))

    def clear_paths(self) -> None:
        for p in self._paths:
            p.__del__()
        self._paths.clear()
        
    def clear_path(self, pos: int) -> None:
        if pos < len(self._paths):
            self._paths[pos].__del__()
            self._paths[pos] = None


    def clear_coordinates(self) -> None:
        for p in self._poses:
            p.__del__()
        self._poses.clear()
        
    def clear_coordinate(self, pos: int) -> None:
        if pos < len(self._poses):
            self._poses[pos].__del__()
            self._poses[pos] = None
            
    def __del__(self) -> None:
        self.clear_objects()
        self.clear_paths()
        self._session = None
        
    def reset(self) -> None:
        self.clear_objects()
        self.clear_paths()
        self.clear_coordinates()
        self._session.world = self._session.client.get_world()
        current_map_name = self._session.world.get_map().name
        last_map_name = current_map_name.split('/')[-1]
        self._session.client.load_world(last_map_name)
        
