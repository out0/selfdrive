
from pydriveless import Waypoint, angle
from pydriveless import CoordinateConverter
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlanningResult, PlannerResultType
from .. model.planning_data import PlanningData
import math
from scipy import interpolate

HALF_PI = 0.5*math.pi

class Interpolator(LocalPlannerExecutor):
    _map_coordinate_converter: CoordinateConverter

    def __init__(self, map_coordinate_converter: CoordinateConverter,
                 max_exec_time_ms: int):
        LocalPlannerExecutor.__init__(self, "Interpolator", max_exec_time_ms)
        self._map_coordinate_converter = map_coordinate_converter

    def __dedup_path(self, path: list[Waypoint], og_height: int):
        dedup = set()
        new_path = []
        for p in path:
            k = og_height * p.z + p.x
            if k in dedup:
                continue
            dedup.add(k)
            new_path.append(p)
        return new_path

    def _loop_plan(self, planning_data: PlanningData) -> bool:
        start = planning_data.start()

        l1 = planning_data.local_goal()

        path: list[Waypoint] = None

        path = Interpolator.interpolate_hermite_curve(planning_data.og().width(), planning_data.og().height(), start, l1)

        if path is None:
            return False

        if self._check_timeout():
            return False

        path = self.__dedup_path(path, og_height=planning_data.og().height())
        self._set_planning_result(PlannerResultType.INVALID_PATH, path)

        if not planning_data.og().check_feasible_path(planning_data.min_distance(), path):
            return False

        self._set_planning_result(PlannerResultType.VALID, path)
        return False

    def _loop_optimize(self, planning_data: PlanningData) -> bool:
        # ignore
        return False

    def __sort_partition(ref_lst: list[int], data_list: list[int], start: int, end: int):
        pivot = ref_lst[start]

        left_pos = start+1
        right_pos = end

        done = False
        while not done:

            while left_pos <= right_pos and ref_lst[left_pos] <= pivot:
                left_pos = left_pos + 1

            while ref_lst[right_pos] >= pivot and right_pos >= left_pos:
                right_pos = right_pos - 1

            if right_pos < left_pos:
                done = True
            else:
                temp = ref_lst[left_pos]
                ref_lst[left_pos] = ref_lst[right_pos]
                ref_lst[right_pos] = temp

                temp = data_list[left_pos]
                data_list[left_pos] = data_list[right_pos]
                data_list[right_pos] = temp

        temp = ref_lst[start]
        ref_lst[start] = ref_lst[right_pos]
        ref_lst[right_pos] = temp

        temp = data_list[start]
        data_list[start] = data_list[right_pos]
        data_list[right_pos] = temp

        return right_pos

    def __sort_points(ref_lst: list[int], data_list: list[int]):
        Interpolator.__sort_points_qs(ref_lst, data_list, 0, len(ref_lst) - 1)

    def __sort_points_qs(ref_list: list[int], data_list: list[int], start: int, end: int):
        if start >= end:
            return
        pivot = Interpolator.__sort_partition(ref_list, data_list, start, end)
        Interpolator.__sort_points_qs(ref_list, data_list, start, pivot-1)
        Interpolator.__sort_points_qs(ref_list, data_list, pivot + 1, end)

    def cubic_spline_interpolation(p1: Waypoint, l1: Waypoint, l2: Waypoint, max_dist: int) -> list[Waypoint]:
        res_z = [
            p1.z,
            l1.z,
            l2.z
        ]
        res_x = [
            p1.x,
            l1.x,
            l2.x
        ]
        Interpolator.__sort_points(res_z, res_x)

        # Cubic spline
        tck = interpolate.splrep(res_z, res_x, k=2)
        z = p1.z
        forward_movement = z > l1.z
        path_candidate: list[Waypoint] = []

        if forward_movement:
            while z > l1.z and z >= 0:
                p = interpolate.splev(z, tck)
                if p is None:
                    continue
                x = math.floor(p)
                path_candidate.append(Waypoint(x, z))
                z -= 1
        else:
            while z < l1.z and z < max_dist:
                p = interpolate.splev(z, tck)
                if p is None:
                    continue
                x = math.floor(p)
                path_candidate.append(Waypoint(x, z))
                z += 1

        return path_candidate

    def interpolate_hermite_curve(width, height, p1: Waypoint, p2: Waypoint):
        curve = []

        dx = p2.x - p1.x
        dz = p2.z - p1.z
        d = math.hypot(dx, dz)

        a1 = p1.heading.rad() - math.pi / 2
        a2 = p2.heading.rad() - math.pi / 2

        # Tangent vectors
        tan1 = (d * math.cos(a1), d * math.sin(a1))
        tan2 = (d * math.cos(a2), d * math.sin(a2))

        num_points = int(math.hypot(dx, dz))
        if num_points < 2:
            num_points = 2

        last_x = -1
        last_z = -1

        for i in range(num_points):
            t = i / (num_points - 1)
            t2 = t * t
            t3 = t2 * t

            # Hermite basis functions
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            x = h00 * p1.x + h10 * tan1[0] + h01 * p2.x + h11 * tan2[0]
            z = h00 * p1.z + h10 * tan1[1] + h01 * p2.z + h11 * tan2[1]

            if x < 0 or x >= width:
                continue
            if z < 0 or z >= height:
                continue

            cx = int(round(x))
            cz = int(round(z))

            if cx == last_x and cz == last_z:
                continue
            if cx < 0 or cx >= width:
                continue
            if cz < 0 or cz >= height:
                continue

            t00 = 6 * t2 - 6 * t
            t10 = 3 * t2 - 4 * t + 1
            t01 = -6 * t2 + 6 * t
            t11 = 3 * t2 - 2 * t

            ddx = t00 * p1.x + t10 * tan1[0] + t01 * p2.x + t11 * tan2[0]
            ddz = t00 * p1.z + t10 * tan1[1] + t01 * p2.z + t11 * tan2[1]

            heading = HALF_PI - math.atan2(-ddz, ddx) 

            # Interpolated point
            curve.append(Waypoint(cx, cz, angle.new_rad(heading)))
            last_x = cx
            last_z = cz

        return curve
