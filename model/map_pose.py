import math

class MapPose:
    x: float
    y: float
    z: float
    heading: float

    def __init__(self, x: float, y: float, z: float, heading: float):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
    
    def distance_between(p1 : 'MapPose', p2 : 'MapPose') -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx**2 + dy**2)
    
    def dot(p1 : 'MapPose', p2 : 'MapPose') -> float:
        return p1.x * p2.x + p1.y * p2.y

    def from_str(payload: str) -> 'MapPose':
        p = payload.split("|")
        return MapPose(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3])
        )

    def __str__(self) -> str:
        return f"{self.x}|{self.y}|{self.z}|{self.heading}"
