#
# This class tries its best to find a good and feasible goal local WayPoint when given the
# local VehiclePose.
#
import math, numpy as np
from model.waypoint import Waypoint
from model.map_pose import MapPose
from typing import Union, Tuple
from model.physical_parameters import PhysicalParameters
from utils.cudac.cuda_frame import GridDirection
from vision.occupancy_grid_cuda import OccupancyGrid
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


TOO_FAR_THRESHOLD = 200


class GoalPointDiscover:

    _map_coordinate_converter: CoordinateConverter
    _ego_start: Waypoint
    _ego_upper_bound: Waypoint
    _ego_lower_bound: Waypoint

    def __init__(self,
                 map_coordinate_converter: CoordinateConverter) -> None:
        self._map_coordinate_converter = map_coordinate_converter
        self._ego_start = None


    def find_goal(self, og: OccupancyGrid, current_pose: MapPose, goal_pose: MapPose) -> GoalPointDiscoverResult:

        goal = self._map_coordinate_converter.convert_to_waypoint(current_pose, goal_pose)

        if self._ego_start is None:
            self._ego_start = Waypoint(
                og.width() / 2, self._ego_upper_bound.z - 1)

        if self._check_goal_is_in_range_and_feasible(og, goal):
            return GoalPointDiscoverResult(og, self._ego_start, goal, 0, self._check_too_close(og, goal))
        

        return self._find_local_goal_for_out_of_range_goal(og, goal)

    def _find_local_goal_for_out_of_range_goal(self, og: OccupancyGrid, out_of_range_goal: Waypoint):

        dist = Waypoint.compute_euclidian_distance(self._ego_start, out_of_range_goal)
        direction = self._compute_direction(self._ego_start, out_of_range_goal)

        if dist >= TOO_FAR_THRESHOLD:
            direction = self._degenerate_direction(direction)

        og.set_goal_vectorized(out_of_range_goal)
        
        # boundaries means that the goal is feasible a waypoint at the border of the BEV
        goal = self._find_best_goal_on_boundaries(og, out_of_range_goal, direction, BORDER_CHECK_DEPTH)
        if goal is not None:
            return GoalPointDiscoverResult(og, self._ego_start, goal, direction, self._check_too_close(og, goal))
         
        goal = self._find_any_feasible_in_direction(og, out_of_range_goal, direction)
        return GoalPointDiscoverResult(og, self._ego_start, goal, direction, self._check_too_close(og, goal))

    
    
    
    
### testing ignored for now
    
    def __find_best_goal_on_boundary_BOTTOM_LEFT(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
        width = og.width()
        height = og.height()
        min_dist_z = og.get_minimal_distance_z()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(0, round(width/2)),
            x_range_rows=(0, border_depth),
            z_range= (round(height/2 + 2*min_dist_z), height - 1),
            z_range_cols= (0, border_depth),
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0 | HEADING_MINUS_45)
        )
        return goal
    
    def __find_best_goal_on_boundary_BOTTOM_RIGHT(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
        width = og.width()
        height = og.height()
        min_dist_z = og.get_minimal_distance_z()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(round(width/2), width - 1),
            x_range_rows=(0, border_depth),
            z_range= (round(height/2 + 2*min_dist_z), height - 1),
            z_range_cols= (width - border_depth - 1, width - 1),
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0 | HEADING_45)
        )
        return goal
    
    def __find_best_goal_on_boundary_BOTTOM(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
        width = og.width()
        height = og.height()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(0, width - 1),
            x_range_rows=(height - border_depth - 1, height - 1),
            z_range=None,
            z_range_cols=None,
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0)
        )
        return goal

### tested as side-effect of other methods

    def __find_best_goal_heading(self, frame_direction: int, relative_heading: float):
        if frame_direction & HEADING_FROM_START > 0:
            return relative_heading

        if frame_direction & TOP > 0 and frame_direction & LEFT > 0:
                return math.radians(-135)
        elif frame_direction & TOP > 0 and frame_direction & RIGHT > 0:
            return math.radians(-45)
        elif frame_direction & BOTTOM > 0 and frame_direction & LEFT > 0:
            return math.radians(135)
        elif frame_direction & BOTTOM > 0 and frame_direction & RIGHT > 0:
            return math.radians(45)
        elif frame_direction & TOP > 0:
            return math.radians(-90)
        elif frame_direction & BOTTOM > 0:
            return math.radians(90)
        elif frame_direction & RIGHT > 0:
            return math.radians(0)
        elif frame_direction & LEFT > 0:
            return math.radians(180)
        else:
            return 0
 

    ### TESTED
    def _find_best_goal_on_boundaries(self, og: OccupancyGrid, out_of_range_goal: Waypoint, direction: int, border_depth: int) -> Waypoint:

        relative_heading = OccupancyGrid.compute_heading(self._ego_start, out_of_range_goal)
        goal = None

        if direction & TOP > 0 and direction & LEFT > 0:
            goal = self._find_best_goal_on_boundary_TOP_LEFT(og, relative_heading, border_depth)
            if goal is None:
                return self._find_best_goal_on_boundaries(og, out_of_range_goal, TOP, border_depth)
            return goal
            
        elif direction & TOP > 0 and direction & RIGHT > 0:
            goal = self._find_best_goal_on_boundary_TOP_RIGHT(og, relative_heading, border_depth)
            if goal is None:
                return self._find_best_goal_on_boundaries(og, out_of_range_goal, TOP, border_depth)
            return goal
                
        elif direction & TOP > 0:
            return self._find_best_goal_on_boundary_TOP(og, relative_heading, border_depth)
        
        elif direction & BOTTOM > 0 and direction & LEFT > 0:
            goal = self.__find_best_goal_on_boundary_BOTTOM_LEFT(og, relative_heading, border_depth)
            if goal is None:
                return self._find_best_goal_on_boundaries(og, out_of_range_goal, BOTTOM, border_depth)
            return goal
        
        elif direction & BOTTOM > 0 and direction & RIGHT > 0:
            goal = self.__find_best_goal_on_boundary_BOTTOM_RIGHT(og, relative_heading, border_depth)
            if goal is None:
                return self._find_best_goal_on_boundaries(og, out_of_range_goal, BOTTOM, border_depth)
            return goal
        
        elif direction & BOTTOM:
            return self.__find_best_goal_on_boundary_BOTTOM(og, relative_heading, border_depth)
        
        return None
    
    ### TESTED
    def _compute_direction(self, start: Waypoint, goal: Waypoint) -> int:
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
    
    ### TESTED
    def _degenerate_direction(self, direction: int) -> int:
        direction &= ~ LEFT
        direction &= ~ RIGHT
        
        if direction == 0:
            return TOP
        
        return direction

    ### TESTED
    def _find_best_goal_in_range(self, 
                                    frame: np.ndarray, 
                                    x_range: tuple[int, int],
                                    x_range_rows: tuple[int, int],
                                    z_range: tuple[int, int],
                                    z_range_cols: tuple[int, int],
                                    relative_heading: float,
                                    allowed_directions: int) -> tuple[Waypoint, float]:
        
        best_goal = None
        best_goal_cost = 9999999
        
        
        if x_range is not None or x_range_rows is not None:
            min, max = x_range
            row_s, row_e = x_range_rows
            for z in range(row_s, row_e + 1):
                for x in range(min, max + 1):
                    frame_direction = round(frame[z, x, 2])
                    goal_cost = frame[z, x, 1]
                    
                    if frame_direction & allowed_directions > 0 and best_goal_cost > goal_cost:
                        best_heading = self.__find_best_goal_heading(frame_direction, relative_heading)
                        best_goal = Waypoint(x, z, best_heading)
                        best_goal_cost = goal_cost
        
        
        if z_range is None or z_range_cols is None:
            return best_goal, best_goal_cost
        
        
        min, max = z_range
        col_s, col_e = z_range_cols
        for x in range(col_s, col_e + 1):
            for z in range(min, max + 1):
                
                frame_direction = round(frame[z, x, 2])
                goal_cost = frame[z, x, 1]
                
                if frame_direction & allowed_directions > 0 and best_goal_cost > goal_cost:
                    best_heading = self.__find_best_goal_heading(frame_direction, relative_heading)
                    best_goal = Waypoint(x, z, best_heading)
                    best_goal_cost = goal_cost
            
        return best_goal, best_goal_cost

    ### TESTED
    def _check_goal_is_in_range_and_feasible(self, og: OccupancyGrid, goal: Waypoint) -> bool:
        if goal.x < 0 or goal.x >= og.width() or goal.z < 0 or goal.z >= og.height():
            return False

        goal.heading = OccupancyGrid.compute_heading(self._ego_start, goal)
        return og.check_waypoint_feasible(goal)
    
    ### TESTED
    def _check_too_close(self, og: OccupancyGrid, goal: Waypoint) -> bool:
        if goal is None:
            return False
        dx = goal.x - self._ego_start.x
        dz = goal.z - self._ego_start.z
        dist = math.sqrt(dx ** 2 + dz ** 2)
        return dist <= 2 * max(og.get_minimal_distance_x(), og.get_minimal_distance_z())
    
    
    ### TESTED
    def _find_best_goal_on_boundary_TOP_LEFT(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
   
        width = og.width()
        height = og.height()
        min_dist_z = og.get_minimal_distance_z()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(0, round(width/2)),
            x_range_rows=(0, border_depth),
            z_range=( 0, round(height/2 - 2 * min_dist_z)),
            z_range_cols= (0, border_depth),
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0 | HEADING_45)
        )
        return goal

    ### TESTED
    def _find_best_goal_on_boundary_TOP_RIGHT(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
   
        width = og.width()
        height = og.height()
        min_dist_z = og.get_minimal_distance_z()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(round(width/2), width - 1),
            x_range_rows=(0, border_depth),
            z_range=( 0, round(height/2 - 2 * min_dist_z)),
            z_range_cols= (width - border_depth - 1, width - 1),
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0 | HEADING_MINUS_45)
        )
        return goal

    ### TESTED
    def _find_best_goal_on_boundary_TOP(self, 
                                            og: OccupancyGrid, 
                                            relative_heading: float, 
                                            border_depth: int) -> Waypoint:
        width = og.width()
   
        goal, _ = self._find_best_goal_in_range(
            frame=og.get_frame(), 
            x_range=(0, width - 1),
            x_range_rows=(0, border_depth),
            z_range=None,
            z_range_cols=None,
            relative_heading=relative_heading,
            allowed_directions=(HEADING_FROM_START | HEADING_0)
        )
        return goal
    
    ### TESTED
    def _find_any_feasible_in_direction(self, og: OccupancyGrid, out_of_range_goal: Waypoint, direction: int) -> Waypoint:
        
        width = og.width()
        height = og.height()
        min_dist_z = og.get_minimal_distance_z()
        relative_heading = OccupancyGrid.compute_heading(self._ego_start, out_of_range_goal)
   
        if direction & TOP > 0:
   
            goal, _ = self._find_best_goal_in_range(
                frame=og.get_frame(), 
                x_range=(0, width - 1),
                x_range_rows=(0, round(height/2 - 2 * min_dist_z)),
                z_range=None,
                z_range_cols=None,
                relative_heading=relative_heading,
                allowed_directions=(HEADING_FROM_START | HEADING_0)
            )
            return goal
        else:
            goal, _ = self._find_best_goal_in_range(
                frame=og.get_frame(), 
                x_range=(0, width - 1),
                x_range_rows=(round(height/2 + 2 * min_dist_z), height - 1 ),
                z_range=None,
                z_range_cols=None,
                relative_heading=relative_heading,
                allowed_directions=(HEADING_FROM_START | HEADING_0)
            )
            return goal