# https://math.stackexchange.com/questions/1369155/how-do-i-rate-smoothness-of-discretely-sampled-data-picture

import numpy as np
from model.waypoint import Waypoint
from model.physical_parameters import PhysicalParameters

class Jerk2D:
    
    def __compute_path_length(path: list[Waypoint]) -> float:
        len = 0        
        last = path[0]        
        for i in range(1, len(path)):
            curr = path[i]
            len += Waypoint.distance_between(curr, last)
    
    @classmethod
    def compute_path_jerk(cls, path: list[Waypoint], velocity: float):
        total_time = PhysicalParameters.OG_REAL_HEIGHT / velocity
        dt = total_time / len(path)
        
        x = []
        z = []
        for p in path:
            x.append(p.x)
            z.append(p.z)
        
    # Compute first, second, and third derivatives using finite differences
        dx = np.gradient(x, dt)
        dz = np.gradient(z, dt)
        
        ddx = np.gradient(dx, dt)
        ddz = np.gradient(dz, dt)
        
        dddx = np.gradient(ddx, dt)
        dddx = np.gradient(ddz, dt)
        
        # Compute jerk magnitude
        jerk_magnitude = np.sqrt(dddx**2 + ddz**2)
        
        # Total jerk is the sum of jerk magnitudes multiplied by dt (discrete approximation of the integral)
        total_jerk = np.sum(jerk_magnitude) * dt
        
        return total_jerk / PhysicalParameters.OG_REAL_HEIGHT