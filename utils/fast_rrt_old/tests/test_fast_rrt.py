import cv2
import sys
sys.path.append("../../../")
from utils.cudac.cuda_frame import CudaFrame
from model.physical_parameters import PhysicalParameters
from model.waypoint import Waypoint
from utils.fast_rrt.fast_rrt import FastRRT


def main():
    frame = cv2.imread("bev_1.png")
    cuda_frame = CudaFrame(
        frame,
        PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
        PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
        PhysicalParameters.EGO_LOWER_BOUND,
        PhysicalParameters.EGO_UPPER_BOUND
    )
    
    rrt = FastRRT()
    
    start = Waypoint(128, 128, 0.0)
    end = Waypoint(115, 0, 0.6930603027112467)
    
    rrt.set_plan_data(cuda_frame, start, end, 1.0)
    rrt.search()
    
    path = rrt.get_path()
    print (f"the planned path has {len(path)} points")


if __name__ == "__main__":
    main()
