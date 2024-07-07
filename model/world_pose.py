import math

class WorldPose:    
    lat: float  # degrees
    lon: float  # degrees
    alt: float  # meters
    heading: float  # degrees
    
    EARTH_RADIUS = 6378135 # meters

    def __init__(self, lat: float, lon: float, alt: float, heading: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.heading = heading
    
    # O(logn)
    def distance_between(p1 : 'WorldPose', p2 : 'WorldPose') -> float:
        """Compute the Haversine distance between two world absolute poses (ignoring heading)

        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            float: distance in meters
        """
        lat1 = math.radians(p1.lat)
        lat2 = math.radians(p2.lat)
        lon1 = math.radians(p1.lon)
        lon2 = math.radians(p2.lon)
        
        dLat = (lat1 - lat2)
        dLon = (lon1- lon2) 
        a = 0.5 - math.cos(dLat) / 2 + math.cos(lat2) * math.cos(lat1) * (1 - math.cos(dLon)) / 2
        return 6371000 * 2 * math.asin(math.sqrt(a))
                

    def compute_heading(p1 : 'WorldPose', p2 : 'WorldPose') -> float:
        """Computes the bearing (World heading or forward Azimuth) for two world poses

        Args:
            p1 (WorldPose): origin (lat, lon)
            p2 (WorldPose): dest (lat, lon)

        Returns:
            float: angle in radians
        """
        
        lat1 = math.radians(p1.lat)
        lat2 = math.radians(p2.lat)
        lon1 = math.radians(p1.lon)
        lon2 = math.radians(p2.lon)
        
        y = math.sin(lon2 - lon1) * math.cos(lat2)
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)
        theta = math.atan2(y, x)
        p = 2*math.pi
        return (theta + p) % p

    def from_str(payload: str) -> 'WorldPose':
        p = payload.split("|")
        return WorldPose(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(p[3])
        )

    def __str__(self) -> str:
        return f"{self.lat}|{self.lon}|{self.alt}|{self.heading}"
