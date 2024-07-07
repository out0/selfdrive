

class IMUData:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    heading: float
    
    def __init__(self) -> None:
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.heading = 0.0

class GpsData:
    latitude: float
    longitude: float
    altitude: float
    
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt

class Gps:
    def read(self) -> GpsData:
        pass

class Odometer:
    def read(self) -> float:
        pass

class IMU:
    def read(self) -> IMUData:
        pass