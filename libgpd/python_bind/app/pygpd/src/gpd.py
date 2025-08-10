import ctypes, os
from pydriveless import SearchFrame, Waypoint, angle, CoordinateConverter, MapPose



class GoalPointDiscover:
    """
    CUDA Goal point discover 
    """
    __coordinate_converter: CoordinateConverter
    __proximity_threshold_px: int
    
    def __init__(self, coordinate_converter: CoordinateConverter, proximity_threshold_px: int = 250):
        GoalPointDiscover.setup_cpp_lib()
        self.__coordinate_converter = coordinate_converter
        self.__proximity_threshold_px = proximity_threshold_px
        pass

    @classmethod
    def setup_cpp_lib(cls) -> None:
        if hasattr(GoalPointDiscover, "lib"):
            return
        
        lib_path = os.path.join(os.path.dirname(__file__), "../cpp", "libgpd.so")

        GoalPointDiscover.lib = ctypes.CDLL(lib_path)

        GoalPointDiscover.lib.find_lowest_cost_waypoint_for_heading.restype = ctypes.POINTER(ctypes.c_float)
        GoalPointDiscover.lib.find_lowest_cost_waypoint_for_heading.argtypes = [
            ctypes.c_void_p,  # searchFrame
            ctypes.c_int,     # minDistX
            ctypes.c_int,     # minDistZ
            ctypes.c_int,     # goalX
            ctypes.c_int,     # goalZ
            ctypes.c_float,   # heading
            ctypes.c_bool     # compute_exclusion_zones
        ]
        
        GoalPointDiscover.lib.find_lowest_cost_waypoint_direct_to_goal.restype = ctypes.POINTER(ctypes.c_float)
        GoalPointDiscover.lib.find_lowest_cost_waypoint_direct_to_goal.argtypes = [
            ctypes.c_void_p,  # searchFrame
            ctypes.c_int,     # minDistX
            ctypes.c_int,     # minDistZ
            ctypes.c_int,     # goalX
            ctypes.c_int,     # goalZ
            ctypes.c_float,    # next_heading
            ctypes.c_bool     # compute_exclusion_zones
        ]
        
        GoalPointDiscover.lib.find_lowest_error_waypoint_to_goal.restype = ctypes.POINTER(ctypes.c_float)
        GoalPointDiscover.lib.find_lowest_error_waypoint_to_goal.argtypes = [
            ctypes.c_void_p,  # searchFrame
            ctypes.c_int,     # minDistX
            ctypes.c_int,     # minDistZ
            ctypes.c_int,     # goalX
            ctypes.c_int,     # goalZ
            ctypes.c_float,   # best_heading
            ctypes.c_bool     # compute_exclusion_zones
        ]
        
        GoalPointDiscover.lib.free_waypoint_res.restype = None
        GoalPointDiscover.lib.free_waypoint_res.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]


    def __find_lowest_cost_waypoint_for_heading_cuda(search_frame: SearchFrame, goal_x: int, goal_z: int, heading: angle, compute_exclusion_zones: bool) -> Waypoint:
        """
        Find the best waypoint based on the search frame and goal.
        """
        
        min_dist = search_frame.get_last_min_distance()
        
        ptr = GoalPointDiscover.lib.find_lowest_cost_waypoint_for_heading(
            search_frame.get_cuda_ptr(),
            min_dist[0],
            min_dist[1],
            goal_x,
            goal_z,
            heading.rad(),
            compute_exclusion_zones)
        
        x = int(ptr[0])
        z = int(ptr[1])
        
        GoalPointDiscover.lib.free_waypoint_res(ptr)
        
        if x == -1 or z == -1:
            return None
        
        return Waypoint(x, z, heading)

    def __find_lowest_cost_waypoint_direct_to_goal_cuda(search_frame: SearchFrame, goal_x: int, goal_z: int, next_heading_rad: float, compute_exclusion_zones: bool) -> Waypoint:
        """
        Find the best waypoint based on the search frame and goal.
        """
        
        min_dist = search_frame.get_last_min_distance()
        
        ptr = GoalPointDiscover.lib.find_lowest_cost_waypoint_direct_to_goal(
            search_frame.get_cuda_ptr(),
            min_dist[0],
            min_dist[1],
            goal_x,
            goal_z,
            next_heading_rad,
            compute_exclusion_zones)
        
        x = int(ptr[0])
        z = int(ptr[1])
        heading = float(ptr[2])
        
        GoalPointDiscover.lib.free_waypoint_res(ptr)
        
        if x == -1 or z == -1:
            return None
        
        return Waypoint(x, z, angle.new_rad(heading))
       
    def __find_lowest_error_waypoint_to_goal_cuda(search_frame: SearchFrame, goal_x: int, goal_z: int, best_heading: float, compute_exclusion_zones: bool) -> Waypoint:
        """
        Find the best waypoint based on the search frame and goal.
        This is a failsafe method that finds the lowest error waypoint to the goal.
        """
        
        min_dist = search_frame.get_last_min_distance()
        
        ptr = GoalPointDiscover.lib.find_lowest_error_waypoint_to_goal(
            search_frame.get_cuda_ptr(),
            min_dist[0],
            min_dist[1],
            goal_x,
            goal_z,
            best_heading,
            compute_exclusion_zones)
        
        x = int(ptr[0])
        z = int(ptr[1])
        heading = float(ptr[2])
        
        GoalPointDiscover.lib.free_waypoint_res(ptr)
        
        if x == -1 or z == -1:
            return None
        
        return Waypoint(x, z, angle.new_rad(heading))
   
    
    def __check_in_range(self, frame: SearchFrame, l1: Waypoint) -> bool:
        return  l1.x >= 0 and l1.x < frame.width() and \
                l1.z >= 0 and l1.z < frame.height()
        
    
    def find(self, frame: SearchFrame, ego_pose: MapPose, g1: MapPose, g2: MapPose, compute_exclusion_zones: bool = False) -> Waypoint:
        L1 = self.__coordinate_converter.convert(ego_pose, g1)
        L2 = None
        start = Waypoint(frame.width() // 2, frame.height() // 2, angle.new_rad(0.0))
        
        if g2 is not None:
            L2 = self.__coordinate_converter.convert(ego_pose, g2)
        
        if self.__check_in_range(frame, L1):
            return self.__find_in_range(frame, L1, L2, compute_exclusion_zones)
        else:
            start = Waypoint(frame.width() // 2, frame.height() // 2, angle.new_rad(0.0))
            distance_to_l1 = Waypoint.distance_between(start, L1)
            
            if distance_to_l1 < self.__proximity_threshold_px:
                return self.__find_in_proximity(frame, L1, L2, compute_exclusion_zones)
            else:
                return self.__find_far_away(frame, start, L1, compute_exclusion_zones)
        
    def __find_in_range(self, frame: SearchFrame, L1: Waypoint, L2: Waypoint, compute_exclusion_zones: bool) -> Waypoint:
        
        # case 1.1
        if frame.is_traversable(L1.x, L1.z, L1.heading, precision_check=True):
            return L1
        
        if L2 is None:
            # if L2 is None, we can only check L1
            return GoalPointDiscover.__find_lowest_error_waypoint_to_goal_cuda(frame, L1.x, L1.z, L1.heading.rad(), compute_exclusion_zones)
        
        # case 1.2
        mid = Waypoint.mid_point(L1, L2)
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, mid.x, mid.z, L1.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # case 1.3
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, L2.x, L2.z, L2.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # failsafe case
        return GoalPointDiscover.__find_lowest_error_waypoint_to_goal_cuda(frame, L2.x, L2.z, L1.heading.rad(), compute_exclusion_zones)

    
    def __find_in_proximity(self, frame: SearchFrame, L1: Waypoint, L2: Waypoint, compute_exclusion_zones: bool) -> Waypoint:
        
        if L2 is None:
            # if L2 is None, we can only check L1
            return GoalPointDiscover.__find_lowest_error_waypoint_to_goal_cuda(frame, L1.x, L1.z, L1.heading.rad(), compute_exclusion_zones)
               
        # case 2.1
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, L2.x, L2.z, L2.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # case 2.2
        mid = Waypoint.mid_point(L1, L2)
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, mid.x, mid.z, L1.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # failsafe case
        return GoalPointDiscover.__find_lowest_error_waypoint_to_goal_cuda(frame, L2.x, L2.z, L1.heading.rad(), compute_exclusion_zones)
    
    def __find_far_away(self, frame: SearchFrame, start: Waypoint, L1: Waypoint, compute_exclusion_zones: bool) -> Waypoint:
        
        # case 3.1
        mid = Waypoint.mid_point(start, L1)
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, mid.x, mid.z, L1.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # case 3.2
        p = GoalPointDiscover.__find_lowest_cost_waypoint_direct_to_goal_cuda(frame, L1.x, L1.z, L1.heading.rad(), compute_exclusion_zones)
        if p is not None: 
            return p
        
        # failsafe case
        return GoalPointDiscover.__find_lowest_error_waypoint_to_goal_cuda(frame, L1.x, L1.z, L1.heading.rad(), compute_exclusion_zones)
    