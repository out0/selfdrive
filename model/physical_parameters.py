from model.waypoint import Waypoint

class PhysicalParameters:
    OG_REAL_WIDTH: float = 34.641016151377535
    OG_REAL_HEIGHT: float = 34.641016151377535
    OG_WIDTH: int = 256
    OG_HEIGHT: int = 256
    
    OG_WIDTH_PX_TO_METERS_RATE: float = OG_REAL_WIDTH / OG_WIDTH
    OG_HEIGHT_PX_TO_METERS_RATE: float = OG_REAL_HEIGHT / OG_HEIGHT
    
    MIN_DISTANCE_WIDTH_M: float = 3
    MIN_DISTANCE_HEIGHT_M: float = 6
    
    MIN_DISTANCE_WIDTH_PX: int = 22
    MIN_DISTANCE_HEIGHT_PX: int = 40

    EGO_LOWER_BOUND: Waypoint = Waypoint(119, 148) 
    EGO_UPPER_BOUND: Waypoint =  Waypoint(137, 108)

    MAX_STEERING_ANGLE: int = 40
    
    VEHICLE_LENGTH_M: float = 5.412658774  # num px * (OG_REAL_HEIGHT / OG_HEIGHT)