from model.map_pose import MapPose
import numpy as np


class PrePlanningData:
    top_frame:  np.ndarray
    bottom_frame: np.ndarray
    left_frame: np.ndarray
    right_frame: np.ndarray
    pose: MapPose
    velocity: float

    def __init__(self, pose: MapPose, velocity: float, top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
        self.pose = pose
        self.velocity = velocity
        self.top_frame = top
        self.bottom_frame = bottom
        self.left_frame = left
        self.right_frame = right

    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"pose:{self.pose},velocity:{self.velocity},top:{self.__frame_shape_to_str(self.top_frame)},left:{self.__frame_shape_to_str(self.left_frame)},right:{self.__frame_shape_to_str(self.right_frame)},bottom:{self.__frame_shape_to_str(self.bottom_frame)}"

class PlanningData:
    bev: np.ndarray
    pose: MapPose
    velocity: float

    def __init__(self, bev: np.ndarray, pose: MapPose, velocity: float) -> None:
        self.bev = bev
        self.pose = pose
        self.velocity = velocity

    def __frame_shape_to_str(self, frame: np.ndarray) -> str:
        if frame is None:
            return "()"
        return f"({frame.shape[0]},{frame.shape[1]},{frame.shape[2]})"

    def __str__(self) -> str:
        return f"pose:{self.pose},velocity:{self.velocity},bev:{self.__frame_shape_to_str(self.bev)}"


class PlanningDataBuilder:

    def build_planning_data(self) -> PlanningData:
        pass
