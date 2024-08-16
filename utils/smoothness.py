# https://math.stackexchange.com/questions/1369155/how-do-i-rate-smoothness-of-discretely-sampled-data-picture

import numpy as np

class Smoothness2D:
    _y: float
    _dy: float
    _d2y: float
    _w: float
    _last_point: tuple[float, float]
    _count: int
    
    def __init__(self):
        self._y = 0
        self._dy = 0
        self._d2y = 0
        self._last_point = [0, 0]
        self._count = 0
        self._w = 0
    
    
    def add_point (self, x: float, y: float):
        if self._count == 0:
            self._last_point[0] = x
            self._last_point[1] = y
            self._count += 1
            return
        
        l = abs(x - self._last_point[0])
        
        if l == 0:
            self._last_point[0] = x
            self._last_point[1] = y            
            return
        
        lsq = l * l
        
        mat = np.array([
            [6, 3*l, -6, 3*l],
            [3*l, 2 * lsq, -3 * l, lsq],
            [-6, -3*l, 6, -3*l],
            [3*l, lsq, -3 * l, 2 * lsq],
        ])
        
        curr_dy = (y - self._last_point[1]) / l
        
        v = np.array([
            self._last_point[1],
            self._dy,
            y,
            curr_dy
        ])
        
        self._dy = curr_dy
        self._last_point[0] = x
        self._last_point[1] = y
        
        self._w += (1/(lsq * l)) * v.T @ mat @ v
        
    def get_cost(self) -> float:
        return self._w
    
    # def add_point (self, x: float, y: float):
    #     if self._count == 0:
    #         self._last_point[0] = x
    #         self._last_point[1] = y
    #         self._count += 1
    #         return
        
    #     l = abs(x - self._last_point[0])
        
    #     if self._count == 1:
    #         self._dy = (y - self._last_point[1]) / l
    #         self._count += 1
    #         return
        
    #     new_dy = (y - self._last_point[1]) / l
        
    #     self._d2y = 2 / (l ** 2) *\
    #         (3 * (self._last_point[1] - y) + l * (self._dy + 2 * new_dy))
             
    #     self._dy = new_dy
        
    #     self._w += l * (self.)