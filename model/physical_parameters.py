from model.waypoint import Waypoint

class PhysicalParameters:
    OG_REAL_WIDTH: float = 34.641016151377535
    OG_REAL_HEIGHT: float = 34.641016151377535
    OG_WIDTH: int = 256
    OG_HEIGHT: int = 256
    
    MIN_DISTANCE_WIDTH_M: float = 3
    MIN_DISTANCE_HEIGHT_M: float = 6
    
    MIN_DISTANCE_WIDTH_PX: int = 22
    MIN_DISTANCE_HEIGHT_PX: int = 40

    EGO_LOWER_BOUND: Waypoint = Waypoint(119, 148) 
    EGO_UPPER_BOUND: Waypoint =  Waypoint(137, 108)

    MAX_STEERING_ANGLE: int = 40
    