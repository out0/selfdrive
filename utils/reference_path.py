from model.map_pose import MapPose
import math

class ReferencePath:

    def __project_on_path(p1: MapPose, p2: MapPose, p: MapPose) -> tuple[MapPose, float, int]:
        
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
        d = MapPose.dot(v, l)

        return MapPose(
            (p1.x + l.x * d),
            (p1.y + l.y * d),
            0, 0
        ), d, path_size

    def __find_mid (p1: MapPose, p2: MapPose) -> tuple[int, int]:
        return (p2.x + p1.x)/2,  (p2.y + p1.y)/2,
    
    def __compute_dist(x1: int, y1: int, x2: int, y2: int) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx ** 2 + dy ** 2)


    def distance_to_line(p1: MapPose, p2: MapPose, p: MapPose) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y

        if dx == 0 and dy == 0:
            return 0
       
        num = dx*(p1.y - p.y) - (p1.x - p.x)*dy
        den = math.sqrt((dx ** 2 + dy ** 2))
        return num / den
    
        
    def __is_in_p1(location: MapPose, p1: MapPose) -> bool:
        return  math.isclose(location.x, p1.x, rel_tol=1e-2) and\
                math.isclose(location.y, p1.y, rel_tol=1e-2)


    def find_first_goal_for_location(poses: list[MapPose], location: MapPose, max_dist: int = 100) -> int:
        best_dist = ReferencePath.__compute_dist(location.x, location.y, poses[0].x, poses[0].y)
        best = -1
       
        if best_dist <= max_dist:
            best = 0
        
        for i in range(1, len(poses)):
            dist = ReferencePath.__compute_dist(location.x, location.y, poses[i].x, poses[i].y)
            if dist < best_dist and dist < max_dist:
                best_dist = dist
                best = i
        
        return best
    
    def find_best_goal_for_location(poses: list[MapPose], location: MapPose, start: int = 0, max_dist: int = 100) -> int:
        n = len(poses)
        best = -1
        best_dist = 999999999
        best_walked_path_ratio = 0
        best_in_middle = False
        
        nearest: MapPose
        
        if start <= 0:
            start = 1
        
        for i in range (start, n):
            if ReferencePath.__is_in_p1(location, poses[i]):
                return i + 1
            
            nearest, proportion, path_size = ReferencePath.__project_on_path(poses[i-1], poses[i], location)
            
            if path_size == 0:
                continue
            
            if nearest is None:
                continue
            
            if proportion >= path_size: # I'm after the path
                continue

            mx, my = ReferencePath.__find_mid(poses[i-1], poses[i])
            
            dist = ReferencePath.__compute_dist(location.x, location.y, mx, my)
            
            if (dist > max_dist):
                continue
            
            if dist < best_dist:
                best_dist = dist
                best = i
                best_in_middle = proportion > 0
                if best_in_middle:
                    best_walked_path_ratio = proportion / path_size
        
        if best_in_middle and best_walked_path_ratio >= 0.7:
            best = best + 1

        return best
        

    def find_best_p1_for_location(poses: list[MapPose], location: MapPose, start: int = 0, max_dist: int = 100) -> int:
        n = len(poses)
        best = -1
        best_dist = 999999999
        best_walked_path_ratio = 0
        best_in_middle = False
        
        nearest: MapPose
        
        if start == n - 1:
            _, proportion, path_size = ReferencePath.__project_on_path(poses[start - 1], poses[start], location)
            if proportion >= path_size:
                return -1
            return start
                
        for i in range (start, n - 1):
            
            if ReferencePath.__is_in_p1(location, poses[i]):
                return i + 1
            
            nearest, proportion, path_size = ReferencePath.__project_on_path(poses[i], poses[i+1], location)
            
            if path_size == 0:
                continue
            
            if nearest is None:
                continue
            
            if proportion >= path_size: # I'm after the path
                continue

            mx, my = ReferencePath.__find_mid(poses[i], poses[i+1])
            
            dist = ReferencePath.__compute_dist(location.x, location.y, mx, my)
            
            if (dist > max_dist):
                continue
            
            if dist < best_dist:
                best_dist = dist
                best = i
                best_in_middle = proportion > 0
                if best_in_middle:
                    best_walked_path_ratio = proportion / path_size
        
        if best_in_middle and best_walked_path_ratio >= 0.7:
            best = best + 1
        
        if best == len(poses) - 1:
            return -1
        
        return best

