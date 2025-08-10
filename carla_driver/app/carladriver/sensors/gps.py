from .base_sensor import CarlaSensor
from .. control.session import CarlaSession
from pydriveless import GPS, GpsData
   

class CarlaGPS (CarlaSensor, GPS):

    def __init__(self, session: CarlaSession, vehicle: any, period_ms: int):
        CarlaSensor.__init__(self, "sensor.other.gnss", session, vehicle, period_ms, pos=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0])
        
    def read(self) -> GpsData:
        raw_data, timestamp = super().read()
        if raw_data is None: 
           return GpsData(0, 0, 0, False, 0.0) 
        return GpsData(raw_data.latitude, raw_data.longitude, raw_data.altitude, True, timestamp)
        

    