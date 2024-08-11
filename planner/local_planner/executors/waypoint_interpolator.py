
from model.waypoint import Waypoint
from typing import List
import math
from scipy import interpolate


class WaypointInterpolator:

    def __downsample_waypoints(waypoints: List[Waypoint], expected_num_points: int) -> tuple[List[int], List[int]]:
        res_x = []
        res_z = []
        
        last_is_present: bool = False

        division = max(1, math.floor (len(waypoints) / expected_num_points))

        i = 0
        for p in waypoints:
            if i % division == 0:
                res_x.append(p.x)
                res_z.append(p.z)
            i += 1
            if i == len(waypoints) - 1:
                last_is_present = True

        if not last_is_present:
            res_x.append(waypoints[len(waypoints) - 1].x)
            res_z.append(waypoints[len(waypoints) - 1].z)
        
        return (res_x, res_z)
    
    def __sort_points(ref_lst: List[int], data_list: List[int]):
        WaypointInterpolator.__sort_points_qs(ref_lst, data_list, 0, len(ref_lst) - 1)

    def __sort_partition(ref_lst: List[int], data_list: List[int], start: int, end: int):
        pivot = ref_lst[start]

        left_pos = start+1
        right_pos = end

        done = False
        while not done:

            while left_pos <= right_pos and ref_lst[left_pos] <= pivot:
                left_pos = left_pos + 1

            while ref_lst[right_pos] >= pivot and right_pos >= left_pos:
                right_pos = right_pos -1

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

    def __sort_points_qs(ref_list: List[int], data_list: List[int], start: int, end: int):
        if start >= end:
            return

        pivot = WaypointInterpolator.__sort_partition(ref_list, data_list, start, end)

        WaypointInterpolator.__sort_points_qs(ref_list, data_list, start, pivot-1)
        WaypointInterpolator.__sort_points_qs(ref_list, data_list, pivot + 1, end)

    def interpolate_straight_line_path(p1: Waypoint, p2: Waypoint, og_height: int) -> List[Waypoint]:

        forward_movement = p1.z > p2.z
        
        dx = p2.x - p1.x 
        dz = p2.z - p1.z
        
        if dx == 0:
            return []
        
        slope = dx / dz

        dz = (2 * dz) / og_height

        z = p1.z

        result: List[Waypoint] = []

        if forward_movement:
            while z > p2.z:
                x = p1.x + (z - p1.z) * slope
                z += dz
                result.append(Waypoint(math.floor(x), math.floor(z)))
        else:
            while z < p2.z:
                x = p1.x + (z - p1.z) * slope
                z += dz
                result.append(Waypoint(math.floor(x), math.floor(z)))
        
        return result

    def path_interpolate(path: List[Waypoint], goal: Waypoint, og_height: int) -> List[Waypoint]:
        size = len(path)

        if size < 2:
            return None
        if size == 2:
            return WaypointInterpolator.interpolate_straight_line_path(path[0], path[1], og_height)
        
        (res_x, res_z) = WaypointInterpolator.__downsample_waypoints(path, 10)

        WaypointInterpolator.__sort_points(res_z, res_x)

        k = 3
        if len(path) < 4:
            k = 2
        tck = interpolate.splrep(res_z, res_x, k=k)


        z = path[0].z
        last_point = path[len(path) - 1]
        forward_movement = z > last_point.z
        path_candidate: List[Waypoint] = []

        if forward_movement:
            while z > goal.z and z > 0:
                p = interpolate.splev(z, tck)
                if p is None:
                    continue
                x = math.floor(p)
                path_candidate.append(Waypoint(x, z))
                z -= 1
        else:
            while z < goal.z and z < og_height:
                p = interpolate.splev(z, tck)
                if p is None:
                    continue
                x = math.floor(p)
                path_candidate.append(Waypoint(x, z))
                z += 1
        
        return path_candidate

    def path_smooth(path: List[Waypoint]) -> List[Waypoint]:

        size = len(path)

        if size < 4:
            return path
        
        (res_x, res_z) = WaypointInterpolator.__downsample_waypoints(path, 10)

        WaypointInterpolator.__sort_points(res_z, res_x)

        k = 3
        if len(path) < 4:
            k = 2
        tck = interpolate.splrep(res_z, res_x, k=k)

        path_candidate: List[Waypoint] = []

        for p in path:
            try:
                x = math.floor(interpolate.splev(p.z, tck))
                path_candidate.append(Waypoint(x, p.z))
            except:
                pass

        return path_candidate

