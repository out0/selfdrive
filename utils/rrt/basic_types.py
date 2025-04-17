int2 = tuple[int, int]
float2 = tuple[float, float]

class PhysicalParameters:
    rate: float2
    inv_rate: float2
    lr: float
    maxSteering_rad: float
    
    def __init__(self, rate: float2, lr: float, maxSteering_rad: float):
        self.rate = rate
        self.inv_rate = (1/rate[0], 1/rate[1])
        self.lr = lr
        self.maxSteering_rad = maxSteering_rad
