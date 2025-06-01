from model.waypoint import Waypoint
import numpy as np, math, json

class CurveData:
    name: str
    curve: list[Waypoint]
    num_points: int
    jerk: float
    total_length: float
    tan_discontinuity: bool
    coarse: bool
    proc_time_ms: float
    timeout: bool
    goal_reached: bool
    last_p: Waypoint
    
    def __init__(self, 
                 name: str,
                 coarse: bool,
                 curve: list[Waypoint],
                 num_points: int,
                 jerk: float,
                 total_length: float,
                 tan_discontinuity: bool,
                 proc_time_ms: float,
                 num_loops: int,
                 timeout: bool,
                 goal_reached: bool,
                 last_p: Waypoint):
        self.name = name
        self.curve = curve
        self.num_points = num_points
        self.jerk = jerk
        self.total_length = total_length
        self.tan_discontinuity = tan_discontinuity
        self.coarse = coarse
        self.proc_time_ms = proc_time_ms
        self.timeout = timeout
        self.num_loops = num_loops
        self.goal_reached = goal_reached
        self.last_p = last_p

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "curve" : [],
            "type": "coarse" if self.coarse else "optim",
            "num_points": self.num_points,
            "jerk": self.jerk,
            "total_length": self.total_length,
            "tan_discontinuity": self.tan_discontinuity,
            "proc_time_ms": self.proc_time_ms,
            "timeout": self.timeout,
            "num_loops": self.num_loops,
            "goal_reached": self.goal_reached
            
        }
        l = len(self.curve)
        for i in range(l):
            data["curve"].append(f"({int(self.curve[i].x)}, {int(self.curve[i].z)}, {math.degrees(self.curve[i].heading):0.2f})")
            
        return f"{json.dumps(data)}\n"
    
    def to_csv_header(self) -> str:
        return "\"name\";\"num_points\"; \"jerk\"; \"total_length\"; \"tan_discontinuity\"; \"coarse\"; \"proc_time_ms\"; \"timeout\"; \"num_loops\";\"goal_reached\"\n"
    
    def to_csv(self) -> str:
        disc = "yes" if self.tan_discontinuity else "no"
        curve_type = "coarse" if self.coarse else "optim"
        timeout = "yes" if self.timeout else "no"
        goal_reached = "yes" if self.goal_reached else "no"
        return f"\"{self.name}\";\"{self.num_points}\"; \"{self.jerk}\"; \"{self.total_length}\"; \"{disc}\"; \"{curve_type}\"; \"{self.proc_time_ms}\" \"{timeout}\"; \"{self.num_loops}\";\"{goal_reached}\"\n"


class CurveAssessment:
    def __compute_curve_length(curve: list[Waypoint]) -> float:
        size: float = 0
        for i in range(1, len(curve)):
            size += Waypoint.distance_between(curve[i-1], curve[i])
        return size

    
    def assess_curve(curve: list[Waypoint], start_heading: float, compute_heading: bool = True) -> CurveData:
        if (len(curve) == 0):
            return None
        return CurveData(
            name=None,
            coarse=False,
            goal_reached=False,
            timeout=False,
            proc_time_ms=0.0,
            num_loops=0,
            curve=curve,
            num_points=len(curve),
            total_length=CurveAssessment.__compute_curve_length(curve),
            jerk=CurveAssessment.__compute_jerk(curve),
            tan_discontinuity=CurveAssessment.__tangential_discontinuity(curve, window_side=4, threshold=15, start_heading=start_heading, compute_heading=compute_heading),
            last_p=curve[-1]
        )

    def __curve_heading(p1: Waypoint, p2:Waypoint) -> tuple[float, bool]:
        if p1.x == p2.x and p1.z == p2.z: return 0.0, False

        dx = p2.x - p1.x
        dz = p2.z - p1.z

        h = (math.pi / 2) - math.atan2(-dz, dx)
        if h > math.pi:
            h = h - 2 * math.pi

        return h, True

    def __curve_mean_heading(curve: np.ndarray, pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        l = len(curve)

        if i < 0:
            j = j - i
            i = 0
        if j >= l:
            i = i - (j - l) - 1
            j = l - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p, valid = CurveAssessment.__curve_heading(curve[k], curve[pos])
            if not valid and debug:
                print (f"not valid for {k}, {pos}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({k}, {pos}) = {math.degrees(p)} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p, valid = CurveAssessment.__curve_heading(curve[pos], curve[k])
            if not valid and debug:
                print (f"not valid for {pos}, {k}")
                continue
            h += p
            count += 1
            if debug:
                print (f"h({pos}, {k}) = {math.degrees(p)} current mean: {h/count}")

        return h/count

    def __tangential_discontinuity(curve: np.ndarray, window_side: int=2, threshold: float = 15, start_heading: float = 0.0, compute_heading: bool = True):
        #a_before = curve_mean_heading(curve, 0, window_side=window_side)
        a_before = start_heading
        threshold_rad = math.radians(threshold)
        for i in range(1, len(curve)):
            if compute_heading:
                a = CurveAssessment.__curve_mean_heading(curve, i, window_side=window_side)
            else:
                a = curve[i, 2]
            if abs(a - a_before) > threshold_rad:
                return True
                #print(f"pos: {i} {(curve[i][0], curve[i][1])} spike: {a_before} -> {a}")
            a_before = a
        return False


    def __derivate_p(p1: Waypoint, p2: Waypoint) -> float:
        dx = p2.x - p1.x
        dz = p2.z - p1.z
        return math.atan2(dz,dx)

    def __derivate_curve_p(curve: list[Waypoint], pos: int, window_side: int = 2, debug = False):
        i = pos - window_side
        j = pos + window_side

        if i < 0:
            j = j - i
            i = 0
        if j >= len(curve):
            i = i - (j - len(curve)) - 1
            j = len(curve) - 1
        

        h = 0; count = 0

        for k in range(i, pos):
            p = CurveAssessment.__derivate_p(curve[k], curve[pos])
            h += p
            count += 1
            if debug:
                print (f"h({k}, {pos}) = {p} current mean: {h/count}")
            
        for k in range(pos+1, j+1):
            p = CurveAssessment.__derivate_p(curve[pos], curve[k])        
            h += p
            count += 1
            if debug:
                print (f"h({pos}, {k}) = {p} current mean: {h/count}")

        return h/count

    def __gradient(curve: list[Waypoint], window_side: int = 2, debug = False):
        dev_curve = []
        for i in range (0, len(curve)):
            dev_curve.append(CurveAssessment.__derivate_curve_p(curve, pos=i, window_side=window_side, debug=debug))
        return dev_curve

    def __sum(vals: list) -> float:
        v = 0
        for i in range (len(vals)):
            v += abs(vals[i])
        return v

    def __compute_jerk(curve):
        v = CurveAssessment.__gradient(curve, window_side=4)
        if len(v) == 0:
            return 0
        a = np.gradient(v, 1)
        j = np.gradient(a, 1)
        return abs(CurveAssessment.__sum(j))


