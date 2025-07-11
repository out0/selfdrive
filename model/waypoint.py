import math

class Waypoint:
    x: int
    z: int
    heading: float
    reverse: bool

    def __init__(self, x: int, z: int, heading: float = 0, reverse: bool = False):
        self.x = int(x)
        self.z = int(z)
        self.heading = heading
        self.reverse = reverse

    def __str__(self):
        return f"({self.x}, {self.z}, {self.heading})"
    
    
    def __eq__(self, other):
        if other is None: return False
        return  self.x == other.x and \
                self.z == other.z and \
                self.heading == other.heading
    
    def from_str(payload: str) -> 'Waypoint':
        if payload == 'None':
            return None
        
        payload = payload.replace("(","").replace(")", "")
        p = payload.split(",")
        return Waypoint(
            int(p[0]),
            int(p[1]),
            float(p[2])
        )
       
    def distance_between(p1: 'Waypoint', p2: 'Waypoint') -> float:
        """Computes the euclidian distance between two waypoints

        Args:
            p1 (Waypoint): origin
            p2 (Waypoint): dest

        Returns:
            float: distance in pixels
        """
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.sqrt(dx * dx + dz * dz)


    def compute_heading(p1: 'Waypoint', p2: 'Waypoint') -> float:
        """Computes the heading between two waypoints relative to the BEV coordinate system (ref. docs/coordinate_systems.txt)

        Args:
            p1 (Waypoint): origin
            p2 (Waypoint): dest

        Returns:
            float: heading in degrees
        """
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        
        if dx == 0 and dz == 0: return 0
        return math.degrees(math.pi/2 - math.atan2(-dz, dx))
    
    @classmethod
    def distance_to_line(cls, line_p1: 'Waypoint', line_p2: 'Waypoint', p: 'Waypoint') -> float:
        dx = line_p2.x - line_p1.x
        dz = line_p2.z - line_p1.z

        if dx == 0 and dz == 0:
            return 0
       
        num = dx*(line_p1.z - p.z) - (line_p1.x - p.x)*dz
        den = math.sqrt((dx ** 2 + dz ** 2))
        return num / den
    
    @classmethod
    def mid_point(cls, p1: 'Waypoint', p2: 'Waypoint') -> 'Waypoint':
        return Waypoint(math.floor((p2.x + p1.x)/2),  math.floor((p2.z + p1.z)/2))
    
    @classmethod
    def clip(cls, p: 'Waypoint', width: int, height: int) -> 'Waypoint':
        res = Waypoint(p.x, p.z, p.heading)
        
        if res.x < 0:
            res.x = 0
        if res.x >= width:
            res.x = width - 1
        if res.z < 0:
            res.z = 0
        if res.z >= height:
            res.z = height - 1            
        return res

    def clone(self) -> 'Waypoint':
        return Waypoint(
            self.x,
            self.z,
            self.heading
        )