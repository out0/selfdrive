import carla
import math, random
from threading import Thread, Lock
import time
from .base_sensor import CarlaSensor
from .. control.session import CarlaSession
from .. control.ego import EgoVehicle
import numpy as np
import weakref
from pydriveless import Telemetry

TELEMETRY = True

class Lidar(CarlaSensor):
    __lidar_obj: any
    
    def __init__(self, 
                 session: CarlaSession,
                 vehicle: any,
                 period_ms: int,
                 pos: tuple[float, float, float] = (0.0, 0.0, 3.0)
                 ):
            
        CarlaSensor.__init__(self,
                    "sensor.lidar.ray_cast", 
                    session, 
                    vehicle, 
                    period_ms, 
                    pos=pos, 
                    rotation=[0.0, 0.0, 0.0],
                    custom_attributes={
                                "channels": str(128),
                                "horizontal_fov": str(360.0),
                                "rotation_frequency": str(100),
                                "range": str(120.0),
                                "dropoff_general_rate": str(0.1),
                                "dropoff_intensity_limit": str(0.1),
                                "dropoff_zero_intensity": str(0.75),
                                "points_per_second": str(56000),
                                })

    
    