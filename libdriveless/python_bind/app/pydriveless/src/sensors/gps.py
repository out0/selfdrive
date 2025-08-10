
class GpsData:
    __lat: float
    __lon: float
    __alt: float
    __valid: bool
    __timestamp: float
    
    def __init__(self, lat: float, lon: float, alt: float, valid: bool, timestamp: float):
        self.__lat = lat
        self.__lon = lon
        self.__alt = alt
        self.__valid = valid
        self.__timestamp = timestamp
        
        
    def __str__(self):
        if self.__valid:
            return f"({self.__lat:.2f}, {self.__lon:.2f}, {self.__alt:.2f}) [{self.__timestamp}]"
        return "(invalid)"
    
    @property
    def lat(self) -> float:
        return self.__lat

    @property
    def lon(self) -> float:
        return self.__lon

    @property
    def alt(self) -> float:
        return self.__alt

    @property
    def valid(self) -> bool:
        return self.__valid

    @property
    def timestamp(self) -> float:
        return self.__timestamp

class GPS:

    def __init__(self, period_ms: int):
        pass
        
    def read(self) -> GpsData:
        pass
        

    