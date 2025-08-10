
class IMUData:
    __accel_x: float  # m/s²
    __accel_y: float  # m/s²
    __accel_z: float  # m/s²
    __gyro_x: float   # rad/s
    __gyro_y: float   # rad/s
    __gyro_z: float   # rad/s
    __compass: float  # rad
    __timestamp: float
    __valid: bool
    
    def __init__(self,
                 accel_x: float,
                 accel_y: float,
                 accel_z: float,
                 gyro_x: float,
                 gyro_y: float,
                 gyro_z: float,
                 compass: float,
                 valid: bool,
                 timestamp: float
                 ) -> None:
            self.__accel_x = accel_x
            self.__accel_y = accel_y
            self.__accel_z = accel_z
            self.__gyro_x = gyro_x
            self.__gyro_y = gyro_y
            self.__gyro_z = gyro_z
            self.__compass = compass
            self.__timestamp = timestamp
            self.__valid = valid
        
    def __str__(self):
        if not self.__valid:
            return "-"
        return f"accel: ({self.__accel_x:.2f}, {self.__accel_y:.2f}, {self.__accel_z:.2f}); gyro: ({self.__gyro_x:.2f}, {self.__gyro_y:.2f}, {self.__gyro_z:.2f}); compass: {self.__compass:.2f} [{self.__timestamp}]"
    
    @property
    def accel_x(self) -> float:
        return self.__accel_x

    @property
    def accel_y(self) -> float:
        return self.__accel_y

    @property
    def accel_z(self) -> float:
        return self.__accel_z

    @property
    def gyro_x(self) -> float:
        return self.__gyro_x

    @property
    def gyro_y(self) -> float:
        return self.__gyro_y

    @property
    def gyro_z(self) -> float:
        return self.__gyro_z

    @property
    def compass(self) -> float:
        return self.__compass

    @property
    def timestamp(self) -> float:
        return self.__timestamp
    
    @property
    def valid(self) -> bool:
        return self.__valid

class IMU:

    def __init__(self, period_ms: int):
        pass
        
    def read(self) -> IMUData:
        pass
    
