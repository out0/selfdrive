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
    __start: Waypoint
    __goal: Waypoint
    __direction: int
    __too_close: bool
    __occupancy_grid: OccupancyGrid

    def __init__(self, og: OccupancyGrid, start: Waypoint, goal: Waypoint, direction: int, too_close: bool) -> None:
        self.__direction = dir
        self.__start = start
        self.__goal = goal
        self.__direction = direction
        self.__too_close = too_close
        self.__occupancy_grid = og
    
    @property
    def start (self) -> Waypoint:
        return self.__start
    
    @property
    def goal (self) -> Waypoint:
        return self.__goal
        
    @property
    def direction (self) -> int:
        return self.__direction
    
    @property
    def too_close (self) -> bool:
        return self.__too_close
    
    @property
    def occupancy_grid (self) -> OccupancyGrid:
        return self.__occupancy_grid


TOP = 8     # 1000
BOTTOM = 4  # 0100
LEFT = 2    # 0010
RIGHT = 1   # 0001
INSIDE = 0  # 0000


TOO_FAR_THRESHOLD = 150


class SearchParameters:
    start: Waypoint
    g1: Waypoint
    g2: Waypoint
    direction: int
    distance_to_goal: float
    
    def __init__(self,
        start: Waypoint,
        goal_candidate: Waypoint,
        next_goal_candidate: Waypoint,
        direction: int,
        distance_to_goal: float
    ) -> None:
        self.start = start
        self.g1 = goal_candidate
        self.g2 = next_goal_candidate
        self.direction = direction
        self.distance_to_goal = distance_to_goal

class GoalPointDiscover:

    _map_coordinate_converter: CoordinateConverter
    _ego_start: Waypoint
    _ego_upper_bound: Waypoint
    _ego_lower_bound: Waypoint

    def __init__(self,
                 map_coordinate_converter: CoordinateConverter) -> None:
        self._map_coordinate_converter = map_coordinate_converter
        self._ego_start = None
        
    def __compute_initial_parameters(self, og: OccupancyGrid, current_pose: MapPose, goal_pose: MapPose, next_goal_pose: MapPose) -> SearchParameters:
        if self._ego_start is None:
            self._ego_start = Waypoint(
                og.width() / 2, PhysicalParameters.EGO_UPPER_BOUND.z - 1)
        
        
        goal_candidate = self._map_coordinate_converter.convert_map_to_waypoint(current_pose, goal_pose)
        
        params = SearchParameters(
            start=self._ego_start,
            goal_candidate=goal_candidate,
            next_goal_candidate=None,
            direction=self.__compute_direction(self._ego_start, goal_candidate),
            distance_to_goal=Waypoint.distance_between(self._ego_start, goal_candidate)
        )
        
        if next_goal_pose is not None:
            params.g2 = self._map_coordinate_converter.convert_map_to_waypoint(current_pose, next_goal_pose)

        return params

    def find_goal(self, og: OccupancyGrid, current_pose: MapPose, goal_pose: MapPose, next_goal_pose: MapPose) -> GoalPointDiscoverResult:

        params = self.__compute_initial_parameters(og, current_pose, goal_pose, next_goal_pose)
        
        if params.direction & BOTTOM:
            return GoalPointDiscoverResult(
                    og=og,
                    start=self._ego_start,
                    goal=None,
                    direction=BOTTOM,
                    #too_close=False
                    too_close=False)

        og.set_goal_vectorized(params.g1)

        goal = self.__try_direct_goal(og, params)
        
        if goal is None:
            goal = self.__find_local_goal_to_reach(og, params, params.g1)
        
            if goal is None: 
                return GoalPointDiscoverResult(
                    og=og,
                    start=self._ego_start,
                    goal=None,
                    direction=0,
                    #too_close=False
                    too_close=False
        )
            
        goal.heading = self.__find_best_alternative_heading_to_goal(og, params, goal)
        direction = self.__compute_direction_from_heading_angle(goal.heading)
        
        return GoalPointDiscoverResult(
            og=og,
            start=self._ego_start,
            goal=goal,
            direction=direction,
            #too_close=False
            too_close=self._check_too_close(og, goal)
        )

    def _check_too_close(self, og: OccupancyGrid, goal: Waypoint) -> bool:
        if goal is None:
            return False
        dx = goal.x - self._ego_start.x
        dz = goal.z - self._ego_start.z
        dist = math.sqrt(dx ** 2 + dz ** 2)
        return dist <= 2 * max(og.get_minimal_distance_x(), og.get_minimal_distance_z())

    # let g1 = goal
    #     g2 = next goal
    
    def __try_projecting_g1(self, og: OccupancyGrid, params: SearchParameters) -> tuple[Waypoint, int]:
        # project g1 locally
        goal_candidate = Waypoint.clip(params.g1, og.width(), og.height())
        # set heading g1 --> g2
                
        goal_candidate.heading = Waypoint.compute_heading(params.g1, params.g2)
        if og.check_waypoint_feasible(goal_candidate):
            return goal_candidate
        
        # if not feasible, then check if we can reach g1 with any heading 
        if og.check_any_direction_allowed(goal_candidate.x, goal_candidate.z):
            goal_candidate.heading = self.__find_best_alternative_heading_to_goal(og, params, goal_candidate)
            return goal_candidate
        
        return None

    def __try_direct_goal(self, og: OccupancyGrid, params: SearchParameters) -> Waypoint:
        
        if self.__goal_in_range(og, params.g1):            
            if params.g2 is None:
                return self.__find_direct_final_goal(og, params)
            else:
                return self.__find_direct_goal_with_next_goal(og, params)
        
        ## G1 is not in range
        
        if params.distance_to_goal > TOO_FAR_THRESHOLD:
            # it is not good to go directly, because g1 is too far
            return None
        
        goal = self.__try_projecting_g1(og, params)
        if goal is not None:
            return goal

        # g1 is not reachable, lets keep searching for alternatives
        return None
        
    def __find_direct_goal_with_next_goal(self, og: OccupancyGrid, params: SearchParameters) -> Waypoint:
        # When g1 is in range and there's a g2, we want to (1) go towards g1 if possible and (2) arrive at g1 with heading g1 ---> g2
        params.g1.heading = Waypoint.compute_heading(params.g1, params.g2)
        if og.check_waypoint_feasible(params.g1):
            return params.g1
                
        # if (1) is not possible, then g1 is not a good choice, we want to go directly to g2
        goal = self.__find_local_goal_to_reach(og, params, params.g2)
        
        if goal is None: 
            return None
            
        goal.heading = self.__find_best_alternative_heading_to_goal(og, params, goal)
        return goal
    
    def __find_direct_final_goal(self, og: OccupancyGrid, params: SearchParameters) -> Waypoint:
        # In case g2 is None, we reached the end of our path, so we just need to reach g1.
        params.g1.heading = Waypoint.compute_heading(params.start , params.g1)
        
        if og.check_waypoint_feasible(params.g1):
            return params.g1
        
        if og.check_any_direction_allowed(params.g1.x, params.g1.z):
            params.g1.heading = self.__find_best_alternative_heading_to_goal(og, params, params.g1)
            return params.g1
        
        return None
    
    def __find_best_alternative_heading_to_goal(self, og: OccupancyGrid, params: SearchParameters, goal: Waypoint) -> float:   
        
        best_theoretical_heading = Waypoint.compute_heading(goal, params.g2)
        
        if goal is not params.g2:
            best_heading = best_theoretical_heading
            
        goal.heading = best_heading
        if og.check_waypoint_feasible(goal):
            return best_heading
        
        best_heading =  Waypoint.compute_heading(params.start, goal)
        allowed_dirs = int(og.get_frame()[goal.z, goal.x, 2])
        heading_error = 99999999
        
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
                
        if params.direction & LEFT:            
            for i in range(len(left_dirs)):
                if not allowed_dirs & left_dirs[i]: continue                
                new_err = abs(dir_angles[i] - abs(best_theoretical_heading))
                if new_err < heading_error:
                    best_heading = -dir_angles[i]
                    heading_error = new_err
        else:
            for i in range(len(right_dirs)):
                if not allowed_dirs & right_dirs[i]: continue                
                new_err =  abs(dir_angles[i] - abs(best_theoretical_heading))
                if new_err < heading_error:
                    best_heading = dir_angles[i]
                    heading_error = new_err
    
        return best_heading
    
    def __find_local_goal_to_reach(self, og: OccupancyGrid, params: SearchParameters, goal_to_reach: Waypoint) -> Waypoint:
        
        
        best_heading = Waypoint.compute_heading(params.g1, params.g2)
        
        distance = Waypoint.distance_between(params.start, goal_to_reach)
        direction = self.__compute_direction(params.start, goal_to_reach)
        
        # if distance > TOO_FAR_THRESHOLD:            
        #     goal = self._find_goal_in_upper_border(og, params.start, direction)
        #     if goal is not None:
        #         return goal

        #     goal = self._find_goal_forward(og, params.start, goal_to_reach, distance)
        #     if goal is not None:
        #         return goal
        
        goal = self._find_best_of_any_goal_in_direction(og, params.start, direction)
        
        if goal is None or self._check_too_close(og, goal):
             return self._find_best_of_any_goal(og, params.start, direction)
                 
        return goal
            
    
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
    
    def _find_best_of_any_goal(self, og: OccupancyGrid,  start: Waypoint, direction: int) -> Waypoint:
        init = Waypoint(0, 0)
        end = Waypoint(og.width() - 1, og.height() - 1)
        
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


    def __compute_direction_from_heading_angle(self, heading: float) -> int:
        
        if heading >= -90 and heading <= 90:
            
            if heading < 0:
                return TOP | LEFT
            elif heading > 0:
                return TOP | RIGHT
            else:
                return TOP
        else:
            if heading < -90:
                return BOTTOM | LEFT
            elif heading == 180:
                return BOTTOM
            else:
                return BOTTOM | RIGHT
            
        