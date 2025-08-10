from pydriveless import Waypoint
import numpy as np

class PhysicalParameters:
    OG_REAL_WIDTH: float = 34.641016151377535
    OG_REAL_HEIGHT: float = 34.641016151377535
    OG_WIDTH: int = 256
    OG_HEIGHT: int = 256
    
    OG_WIDTH_PX_TO_METERS_RATE: float = OG_REAL_WIDTH / OG_WIDTH
    OG_HEIGHT_PX_TO_METERS_RATE: float = OG_REAL_HEIGHT / OG_HEIGHT
    
    MIN_DISTANCE_WIDTH_M: float = 3
    MIN_DISTANCE_HEIGHT_M: float = 6
    
    # MIN_DISTANCE_WIDTH_PX: int = 22
    # MIN_DISTANCE_HEIGHT_PX: int = 40
    
    MIN_DISTANCE_WIDTH_PX: int = 0
    MIN_DISTANCE_HEIGHT_PX: int = 0

    EGO_LOWER_BOUND: tuple[int, int] = (119, 148) 
    EGO_UPPER_BOUND: tuple[int, int] =  (137, 108)

    MAX_STEERING_ANGLE: int = 40

    VEHICLE_LENGTH_M: float = 5.412658774  # num px * (OG_REAL_HEIGHT / OG_HEIGHT)
    
        
    SEGMENTED_COLORS = np.array([
        [0,   0,   0],
        [128,  64, 128],
        [244,  35, 232],
        [70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [0,   0, 142],
        [0,   0,  70],
        [0,  60, 100],
        [0,  80, 100],
        [0,   0, 230],
        [119,  11,  32],
        [110, 190, 160],
        [170, 120,  50],
        [55,  90,  80],     # other
        [45,  60, 150],
        [157, 234,  50],
        [81,   0,  81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180]
    ])

    SEGMENTATION_CLASS_COST = np.array([
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        0,
        0,   # LAMP? investigate...
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1, # car
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        -1
    ], dtype=np.float32)

