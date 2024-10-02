import sys
sys.path.append("../../../")
from utils.cudac.cuda_frame import CudaFrame
import cv2, numpy as np
from model.physical_parameters import PhysicalParameters
import math


raw = np.array(cv2.imread("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive/utils/cudac/unittests/test_data/bev_3.png"), dtype=np.float32)

frame = CudaFrame(
    frame=raw,
    lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
    upper_bound=PhysicalParameters.EGO_UPPER_BOUND,
    min_dist_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
    min_dist_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX
)

p = frame.find_best_cost_waypoint_with_heading(155, 16, 14.56576424655659)

print(p)
