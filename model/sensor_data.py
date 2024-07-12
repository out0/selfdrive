class IMUData:
    accel_x: float  # m/s²
    accel_y: float  # m/s²
    accel_z: float  # m/s²
    gyro_x: float   # rad/s
    gyro_y: float   # rad/s
    gyro_z: float   # rad/s
    compass: float  # rad
    
    def __init__(self) -> None:
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.compass = 0.0
        
class GpsData:
    latitude: float
    longitude: float
    altitude: float
    
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt