from .. control.session import CarlaSession
from .base_sensor import CarlaSensor
from pydriveless import IMU, IMUData


class CarlaIMU (CarlaSensor, IMU):

    def __init__(self, session: CarlaSession, vehicle: any, period_ms: int):
        CarlaSensor.__init__(self,
            "sensor.other.imu", 
            session, 
            vehicle, 
            period_ms, 
            pos=[0.0, 0.0, 0.0], 
            rotation=[0.0, 0.0, 0.0],
            custom_attributes={
                        "noise_accel_stddev_x": 0.0,
                        "noise_accel_stddev_y": 0.0,
                        "noise_accel_stddev_z": 0.0,
                        "noise_gyro_bias_x": 0.0,
                        "noise_gyro_bias_y": 0.0,
                        "noise_gyro_bias_z": 0.0,
                        "noise_gyro_stddev_x": 0.0,
                        "noise_gyro_stddev_y": 0.0,
                        "noise_gyro_stddev_z": 0.0,                        
                        })
        
    def read(self) -> IMUData:
        limits = (-99.9, 99.9)
        
        raw_data, timestamp = super().read()
        if raw_data is None: 
           return IMUData(0, 0, 0, 0, 0, 0, 0, False, timestamp) 
        
        data = IMUData(
            accel_x = max(limits[0], min(limits[1], raw_data.accelerometer.x)),
            accel_y = max(limits[0], min(limits[1], raw_data.accelerometer.y)),
            accel_z = max(limits[0], min(limits[1], raw_data.accelerometer.z)),
            compass = raw_data.compass,
            gyro_x = raw_data.gyroscope.x,
            gyro_y = raw_data.gyroscope.y,
            gyro_z = raw_data.gyroscope.z,
            valid=True,
            timestamp = timestamp
        )
        
        return data
    
