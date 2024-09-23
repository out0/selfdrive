#
# This class tries its best to find a good and feasible goal local WayPoint when given the
# local VehiclePose.
#
import math, numpy as np
from model.waypoint import Waypoint
from model.map_pose import MapPose
from typing import Union, Tuple
from model.physical_parameters import PhysicalParameters
from vision.occupancy_grid_cuda import OccupancyGrid, GridDirection
from data.coordinate_converter import CoordinateConverter

# increase if NOISE in vision is preventing the goal to be found
BORDER_CHECK_DEPTH = 0

class GoalPointDiscoverResult:
    start: Waypoint
    goal: Waypoint
    direction: int
    too_close: bool
    occupancy_grid: OccupancyGrid

    def __init__(self, og: OccupancyGrid, start: Waypoint, goal: Waypoint, direction: int, too_close: bool) -> None:
        self.direction = dir
        self.start = start
        self.goal = goal
        self.direction = direction
        self.too_close = too_close
        self.occupancy_grid = og


TOP = 8     # 1000
BOTTOM = 4  # 0100
LEFT = 2    # 0010
RIGHT = 1   # 0001
INSIDE = 0  # 0000


TOO_FAR_THRESHOLD = 150


class GoalPointDiscover:

    _map_coordinate_converter: CoordinateConverter
    _ego_start: Waypoint
    _ego_upper_bound: Waypoint
    _ego_lower_bound: Waypoint

    def __init__(self,
                 map_coordinate_converter: CoordinateConverter) -> None:
        self._map_coordinate_converter = map_coordinate_converter
        self._ego_start = None


    def find_goal(self, og: OccupancyGrid, current_pose: MapPose, goal_pose: MapPose, next_goal_pose: MapPose) -> GoalPointDiscoverResult:

        goal_candidate = self._map_coordinate_converter.convert_map_to_waypoint(current_pose, goal_pose)
        next_goal_candidate = None
        
        if next_goal_pose is not None:
            next_goal_candidate = self._map_coordinate_converter.convert_map_to_waypoint(current_pose, next_goal_pose)

        if self._ego_start is None:
            self._ego_start = Waypoint(
                og.width() / 2, PhysicalParameters.EGO_UPPER_BOUND.z - 1)

        direction = self.__compute_direction(self._ego_start, goal_candidate)
        distance = Waypoint.distance_between(self._ego_start, goal_candidate)
        
        if direction & BOTTOM:
            return None

        og.set_goal_vectorized(goal_candidate)

        goal = self.__try_direct_goal(og, self._ego_start, goal_candidate, next_goal_candidate,  direction, distance)        
        
        if goal is None:
            goal = self.__find_best_goal(og, self._ego_start, goal_candidate, next_goal_candidate, direction, distance)
            goal.heading = self.__find_best_heading(og, self._ego_start, goal_candidate, next_goal_candidate, direction)
        
        if goal is None: return None        
        
        
        return GoalPointDiscoverResult(
            og=og,
            start=self._ego_start,
            goal=goal,
            direction=direction,
            too_close=False
            #too_close=self._check_too_close(og, goal)
        )

    def _check_too_close(self, og: OccupancyGrid, goal: Waypoint) -> bool:
        if goal is None:
            return False
        dx = goal.x - self._ego_start.x
        dz = goal.z - self._ego_start.z
        dist = math.sqrt(dx ** 2 + dz ** 2)
        return dist <= 2 * max(og.get_minimal_distance_x(), og.get_minimal_distance_z())

    def __try_direct_goal(self, og: OccupancyGrid, start: Waypoint, goal_candidate: Waypoint, next_goal_candidate: Waypoint, direction: int, distance: float) -> Waypoint:
        
        if self.__goal_in_range(og, goal_candidate):            
            if next_goal_candidate != None:
                goal_candidate.heading = Waypoint.compute_heading(goal_candidate, next_goal_candidate)
                if og.check_waypoint_feasible(goal_candidate):
                    return goal_candidate
                else:
                    if og.check_any_direction_allowed(goal_candidate.x, goal_candidate.z):
                        goal_candidate.heading = self.__find_best_heading(og, start, goal_candidate, next_goal_candidate, direction)
                    return goal_candidate

        if distance <= TOO_FAR_THRESHOLD:
            goal = Waypoint.clip(goal_candidate, og.width(), og.height())
            if og.check_any_direction_allowed(goal.x, goal.z):
                goal_candidate.heading = self.__find_best_heading(og, start, goal_candidate, next_goal_candidate, direction)
                return goal

        return None
    
    def __find_best_heading(self, og: OccupancyGrid, start: Waypoint, goal_candidate: Waypoint, next_goal_candidate: Waypoint, direction: int) -> float:   
        goal_candidate.heading = Waypoint.compute_heading(start, goal_candidate)
        best_heading = 999999
        
        if og.check_waypoint_feasible(goal_candidate):
            best_heading = goal_candidate.heading
            
        goal_candidate.heading = Waypoint.compute_heading(goal_candidate, next_goal_candidate)
        if og.check_waypoint_feasible(goal_candidate):
            best_heading = min(best_heading, goal_candidate.heading)
        
        if best_heading < 90:
            return best_heading
        
        allowed_dirs = int(og.get_frame()[goal_candidate.z, goal_candidate.x, 2])
        
        best_heading = goal_candidate.heading
        err = 99999999
        
        left_dirs = [
            GridDirection.HEADING_0.value,
            GridDirection.HEADING_MINUS_22_5.value,
            GridDirection.HEADING_MINUS_45.value,
            GridDirection.HEADING_MINUS_67_5.value,
            GridDirection.HEADING_90.value
        ]
        right_dirs = [
            GridDirection.HEADING_0.value,
            GridDirection.HEADING_22_5.value,
            GridDirection.HEADING_45.value,
            GridDirection.HEADING_67_5.value,
            GridDirection.HEADING_90.value
        ]
        
        dir_angles = [
            0.0,
            22.5,
            45,
            67.5,
            90
        ]
                
        if direction & LEFT:            
            for i in range(len(left_dirs)):
                if not allowed_dirs & left_dirs[i]: continue                
                new_err = dir_angles[i] - abs(goal_candidate.heading)
                if new_err < err:
                    best_heading = -dir_angles[i]
                    err = new_err
        else:
            for i in range(len(right_dirs)):
                if not allowed_dirs & right_dirs[i]: continue                
                new_err = dir_angles[i] - abs(goal_candidate.heading)
                if new_err < err:
                    best_heading = dir_angles[i]
                    err = new_err
    
        return best_heading
    
    def __find_best_goal(self, og: OccupancyGrid, start: Waypoint, goal_candidate: Waypoint, next_goal_candidate: Waypoint, direction: int, distance: float) -> Waypoint:
        if distance > TOO_FAR_THRESHOLD:            
            goal = self._find_goal_in_upper_border(og, start, direction)
            if goal is not None:
                return goal

            goal = self._find_goal_forward(og, start, goal_candidate, distance)
            if goal is not None:
                return goal
        

        return self._find_best_of_any_goal_in_direction(og, goal_candidate, direction)
        
    def _find_goal_in_upper_border(self, og: OccupancyGrid, start: Waypoint, direction: int) -> Waypoint:

        init = Waypoint(0, 0)
        end = Waypoint(0, start.z / 2)

        if direction & RIGHT:
            init.x = og.width() - 1
            end.x = og.width() - 1
            
        return self._find_goal_in_grid(og, init, end)
    
    def _find_goal_forward(self, og: OccupancyGrid, start: Waypoint, goal: Waypoint, distance: float) -> Waypoint:

        heading_to_goal = math.radians(Waypoint.compute_heading(start, goal))
        
        min_z = start.z - math.cos(heading_to_goal) * distance

        if min_z < 0:
            min_z = 0
        
        init = Waypoint(start.x - PhysicalParameters.MIN_DISTANCE_WIDTH_PX, min_z)
        end = Waypoint(start.x + PhysicalParameters.MIN_DISTANCE_WIDTH_PX, start.z)
            
        return self._find_goal_in_grid(og, init, end)
    
    def _find_best_of_any_goal_in_direction(self, og: OccupancyGrid,  start: Waypoint, direction: int) -> Waypoint:
        
        if direction & LEFT:
            init = Waypoint(0, 0)
            end = Waypoint(start.x, start.z)
        elif direction & RIGHT:
            init = Waypoint(start.x, 0)
            end = Waypoint(og.width() - 1, start.z)
        else:
            init = Waypoint(0, 0)
            end = Waypoint(og.width() - 1, start.z)
        
        return self._find_goal_in_grid(og, init, end)
    
    def _find_goal_in_grid(self, og: OccupancyGrid, init: Waypoint, end: Waypoint) -> Waypoint:
        best_goal = None
        best_goal_cost = 9999999
        
        frame = og.get_frame()
                
        for z in range(init.z, end.z + 1):
            for x in range(init.x, end.x + 1):
                if og.check_any_direction_allowed(x, z):
                    cost = frame[z, x, 1]
                    if cost < best_goal_cost:
                        best_goal = Waypoint(x, z)
                        best_goal_cost = cost
        return best_goal
    
    
    
    def __goal_in_range(self, og: OccupancyGrid, goal: Waypoint) -> bool:
        return  goal.x >= 0 and goal.x < og.width() and \
                goal.z >= 0 and goal.z < og.height()

    def __compute_direction(self, start: Waypoint, goal: Waypoint) -> int:
        direction = 0

        # PURE LEFT / RIGHT DOESNT EXIST IN NON-HOLONOMIC SYSTEMS
        if goal.z <= start.z:
            direction = direction | TOP
        elif goal.z > start.z:
            direction = direction | BOTTOM

        if goal.x < start.x:
            direction = direction | LEFT
        elif goal.x > start.x:
            direction = direction | RIGHT

        return direction


