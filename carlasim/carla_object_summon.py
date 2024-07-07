from carlasim.carla_client import CarlaClient
import carla

class CarlaObjectSummoner:
    _client: CarlaClient
    _object_list: list[carla.Actor]
    
    def __init__(self, client: CarlaClient) -> None:
        self._object_list = []
        self._client = client
    
    def __append_and_return(self, obj: carla.Actor) -> carla.Actor:
        self._object_list.append(obj)
        return obj
    
    def summon_object(self, type: str, x: float, y: float, z: float = 2) -> any:
        bp = self._client.get_blueprint(type)
        t = carla.libcarla.Transform( carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(0, 0, 0))
        return self.__append_and_return(self._client.get_world().spawn_actor(bp, t))
        
    def add_cone(self, x: float, y: float, z: float) -> any:
        return self.summon_object(type="static.prop.trafficcone01", x=x, y=y, z=z)
        
    def add_car(self, x: float, y: float, z: float = 2, heading: float = 0, color: str = '144, 238, 144') -> any:
        bp = self._client.get_blueprint("vehicle.tesla.model3")
        bp.set_attribute('color', color)
        t = carla.libcarla.Transform( carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(0, heading, 0))
        controller = carla.VehicleControl()
        controller.brake = 1.0
        c = self.__append_and_return(self._client.get_world().spawn_actor(bp, t))
        c.apply_control(controller)
        return c
        
    def relocate_spectator(self, x: float, y: float, z:float, heading: float, pitch: int = 0) -> None:
        t = carla.libcarla.Transform( carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(pitch, heading, 0))
        self._client.get_world().get_spectator().set_transform(t)
        
    def clear_objects(self) -> None:
        for obj in self._object_list:
            obj.destroy()
        self._object_list.clear()
        
    def show_goal(self, x: float, y: float, z:float) -> None:
        world = self._client.get_world()
        
        world.debug.draw_string(carla.Location(x, y, z), 'x', draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=120000.0,
                                     persistent_lines=True)