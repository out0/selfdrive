from carlasim.carla_client import CarlaClient
import carla
from model.discrete_component import DiscreteComponent
import math

class ExpectatorCameraAutoFollow (DiscreteComponent):
    _client: CarlaClient
    _spectator: any
    _target: any
    _dist_m: float
    
    
    def __init__(self, client: CarlaClient, dist_m: float = 5) -> None:
        super().__init__(50)
        self._client = client
        self._target = None
        self._spectator = None
        self._dist_m = dist_m
            
    def follow(self, obj: any) -> None:
        world = self._client.get_world()
        self._spectator = world.get_spectator()
        self._target = obj
        self.start()
    
    def __compute_vector_xy_components(self, v_size, angle):
        # Convert the angle to radians
        a = math.radians(angle)
        y = v_size * math.sin(a)
        x = v_size * math.cos(a)
        return y, x
    
    
    def _loop(self, dt: float) -> None:
        
        transform = self._target.get_transform()
        heading = transform.rotation.yaw

        y,x = self.__compute_vector_xy_components(self._dist_m, heading)

        spectator_pos = carla.Transform(transform.location + carla.Location(x=-x,y=-y,z=5 ),
                                        carla.Rotation( yaw = heading, pitch = -25))
        self._spectator.set_transform(spectator_pos) 
        
