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
        
        
    def __str__(self) -> str:
        return f"{self.accel_x}|{self.accel_y}|{self.accel_z}|{self.gyro_x}|{self.gyro_y}|{self.gyro_z}|{self.compass}"
    
    def from_str(raw_str: str) -> 'IMUData':
        p = raw_str.split(sep='|')
        data = IMUData()
        data.accel_x = float(p[0])
        data.accel_y = float(p[1])
        data.accel_z = float(p[2])
        data.gyro_x = float(p[3])
        data.gyro_y = float(p[4])
        data.gyro_z = float(p[5])
        data.compass = float(p[6].replace('\n', ''))
    
class GpsData:
    latitude: float
    longitude: float
    altitude: float
    
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt
        
    def __str__(self) -> str:
        return f"{self.latitude}|{self.longitude}|{self.altitude}"
    
    def from_str(raw_str: str) -> 'GpsData':
        p = raw_str.split(sep='|')
        return GpsData(
            lat=float(p[0]),
            lon=float(p[1]),
            alt=float(p[2].replace('\n', ''))
        )
        