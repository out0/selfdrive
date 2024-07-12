from model.world_pose import WorldPose
from model.slam import SLAM


class CarlaSLAM:
    _gps: GPS
    _imu
    _odometer

    def __init__(self, gps, imu, odometer):
        self._gps = gps
        self._imu = imu
        self._odometer = odometer

    def estimate_pose (last_gps: WorldPose, imu_data: IMUData, velocity: float) -> WorldPose:
        pass