from model.world_pose import WorldPose
from model.imu_data import IMUData


class CarlaSLAM:
    _gps
    _imu
    _odometer

    def __init__(self, gps, imu, odometer):
        self._gps = gps
        self._imu = imu
        self._odometer = odometer

    def estimate_pose (last_gps: WorldPose, imu_data: IMUData, velocity: float) -> WorldPose:
        pass