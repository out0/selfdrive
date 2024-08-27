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
        
    def __eq__(self, other: 'MapPose'):
        return  self.x == other.x and\
                self.y == other.y and\
                self.z == other.z and\
                self.heading == other.heading
    
    def __add__(self, other) -> 'MapPose':
        if isinstance(other, MapPose):
            return MapPose (
                self.x + other.x,
                self.y + other.y,
                self.z + other.z,
                self.heading + other.heading
            )
        else:
            return MapPose (
                self.x + other,
                self.y + other,
                self.z + other,
                self.heading
            )
    def __sub__(self, other) -> 'MapPose':
        if isinstance(other, MapPose):
            return MapPose (
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
                self.heading - other.heading
            )
        else:
            return MapPose (
                self.x - other,
                self.y - other,
                self.z - other,
                self.heading
            )


    @classmethod    
    def are_close(cls, p1: 'MapPose', p2: 'MapPose') -> bool:
        return  math.isclose(p1.x, p2.x, rel_tol=1e-2) and\
                math.isclose(p1.y, p2.y, rel_tol=1e-2)
    
    @classmethod
    def distance_between(cls, p1 : 'MapPose', p2 : 'MapPose') -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx**2 + dy**2)
    
    @classmethod
    def dot(cls, p1 : 'MapPose', p2 : 'MapPose') -> float:
        return p1.x * p2.x + p1.y * p2.y

    @classmethod
    def from_str(cls, payload: str) -> 'MapPose':
        if payload == 'None':
            return None
        
        p = payload.split("|")
        return MapPose(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3])
        )


    @classmethod
    def distance_to_line(cls, line_p1: 'MapPose', line_p2: 'MapPose', p: 'MapPose') -> float:
        dx = line_p2.x - line_p1.x
        dy = line_p2.y - line_p1.y

        if dx == 0 and dy == 0:
            return 0
       
        num = dx*(line_p1.y - p.y) - (line_p1.x - p.x)*dy
        den = math.sqrt((dx ** 2 + dy ** 2))
        return num / den

    @classmethod
    def compute_path_heading(cls, p1: 'MapPose', p2: 'MapPose') -> float:
        dy = p2.y - p1.y
        dx = p2.x - p1.x

        if dy >= 0 and dx > 0:                      # Q1
            return math.atan(dy/dx)
        elif dy >= 0 and dx < 0:                    # Q2
            return math.pi - math.atan(dy/abs(dx))
        elif dy < 0 and dx > 0:                     # Q3
            return  -math.atan(abs(dy)/dx)
        elif dy < 0 and dx < 0:                     # Q4
            return math.atan(dy/dx) - math.pi
        elif dx == 0 and dy > 0:
            return  math.pi / 2
        elif dx == 0 and dy < 0:
            return -math.pi / 2
        
        return 0.0


    @classmethod
    def project_on_path(cls, p1: 'MapPose', p2: 'MapPose', p: 'MapPose') -> tuple['MapPose', float, float]:
                
        path_size = MapPose.distance_between(p1, p2)
        
        if path_size == 0:
            return (None, 0, 0)
        
        l = MapPose(
            (p2.x - p1.x) / path_size,
            (p2.y - p1.y) / path_size,
            0, 0
        )
        v = MapPose(
            (p.x - p1.x),
            (p.y - p1.y),
            0, 0
        )
        distance_from_p1 = MapPose.dot(v, l)

        return MapPose(
            round(p1.x + l.x * distance_from_p1),
            round(p1.y + l.y * distance_from_p1),
            0, 0
        ), distance_from_p1, path_size
    
    @classmethod
    def find_nearest_goal_pose(cls, location: 'MapPose', poses: list['MapPose'], start: int = 0) -> int:
        n = len(poses)
        
        if start == n - 1:
            _, distance_from_p1, path_size = MapPose.project_on_path(poses[start - 1], poses[start], location)
            if distance_from_p1 >= path_size:
                # the location is after the end of the line
                return -1
            else:
                return start
        
        return MapPose.__find_nearest_next_pose_from_location(location, poses, start)

    @classmethod
    def __find_mid (cls, p1: 'MapPose', p2: 'MapPose') -> tuple[int, int]:
        return (p2.x + p1.x)/2,  (p2.y + p1.y)/2,
    

    @classmethod
    def __find_nearest_next_pose_from_location(cls, location: 'MapPose', poses: list['MapPose'], start: int = 0, max_dist = 100):
        n = len(poses)
        best = -1
        best_dist = 999999999
        best_walked_path_ratio = 0
        best_in_middle = False
        
        for i in range (start, n - 1):
            
            if MapPose.are_close(location, poses[i]):
                return i+1
            
            _, proportion, path_size = MapPose.project_on_path(poses[i], poses[i+1], location)
            
            if path_size == 0:
                continue
            
            if proportion >= path_size: # I'm after the path
                continue

            mx, my = MapPose.__find_mid(poses[i], poses[i+1])
            
            dist = MapPose.distance_between(location, MapPose(mx, my, 0, 0))
            
            if (dist > max_dist):
                continue
            
            if dist < best_dist:
                best_dist = dist
                best = i + 1
                best_in_middle = proportion > 0
                if best_in_middle:
                    best_walked_path_ratio = proportion / path_size
        
        if best_in_middle and best_walked_path_ratio >= 0.7:
            best = best + 1
        
        if best == n:
            #the best is the last point
            return -1
        
        return best


    def __str__(self) -> str:
        return f"{self.x}|{self.y}|{self.z}|{self.heading}"
