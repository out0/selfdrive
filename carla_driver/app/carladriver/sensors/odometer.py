from pydriveless import Odometer
import math

class CarlaOdometer (Odometer):
    __carla_ego_obj: any
    
    def __init__(self, carla_ego_obj: any):
        super().__init__()
        self.__carla_ego_obj = carla_ego_obj

    def read(self) -> float:
        velocity = self.__carla_ego_obj.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) 
    
