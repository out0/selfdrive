import carla
import math, random
from threading import Thread, Lock
import time
from .. control.session import CarlaSession
from .. control.ego import EgoVehicle
import numpy as np
import weakref
from pydriveless import Telemetry

TELEMETRY = True

class Lidar():
    __lidar_obj: any
    
    def __init__(self, 
                 session: CarlaSession,
                 vehicle_obj: any,
                 pos: tuple[float, float, float] = (0.0, 0.0, 3.0)
                 ):
            self.__timestamp = 0.0
            self.__pos = pos
            lidar_bp = session.world.get_blueprint_library().find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute('channels', str(128))
            lidar_bp.set_attribute('horizontal_fov', str(360.0))
            lidar_bp.set_attribute('rotation_frequency', str(100))
            lidar_bp.set_attribute('range', str(120.0))
            # lidar_bpj.set_attribute("role_name", role_name)
            lidar_bp.set_attribute('dropoff_general_rate', str(0.1))
            lidar_bp.set_attribute('dropoff_intensity_limit', str(0.1))
            lidar_bp.set_attribute('dropoff_zero_intensity', str(0.75))
            lidar_bp.set_attribute('points_per_second', str(56000))

            location = carla.Location(x=self.__pos[0], y=self.__pos[1], z=self.__pos[2])
            rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
            camera_transform = carla.Transform(location, rotation)
            self.__lidar_obj = session.client.get_world().spawn_actor(lidar_bp, camera_transform, attach_to=vehicle_obj)
            #self.__camera_obj.listen(self.test)
            weak_self = weakref.ref(self)
            self.__lidar_obj.listen(lambda p: Lidar.__new_data(weak_self, p))

    def destroy(self):
        self.__lidar_obj.destroy()
        self.__lidar_obj = None

    def __new_data(self, data: carla.LidarMeasurement):
        data.save_to_disk('point_cloud.ply')
        print ("lidar read")