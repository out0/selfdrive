from model.world_pose import WorldPose
from model.sensor_data import IMUData


class ExtKalmanFilterLocation:
    
    def __init__(self) -> None:
        pass

    def estimate_pose (last_gps: WorldPose, imu_data: IMUData, velocity: float) -> WorldPose:
        pass