import carla
from pydriveless import MapPose
from pydriveless import EgoVehicle
from .session import CarlaSession
from .. sensors.gps import CarlaGPS, GpsData
from .. sensors.imu import CarlaIMU, IMUData
from .. sensors.camera import BevCameraSemantic
from .. sensors.odometer import CarlaOdometer

class VehicleState:
    __vehicle: any
    power_level: float
    brake_level: float
    steer_angle: float
    reversed: bool
    
    def __init__(self, vehicle: any):
        self.power_level = 0.0
        self.brake_level = 1.0
        self.steer_angle  = 0.0
        self.reversed = False
        self.__vehicle = vehicle
    
    def apply(self) -> None:
        vehicle_control = carla.VehicleControl()
        vehicle_control.reverse = bool(self.reversed)
        vehicle_control.brake = self.brake_level
        vehicle_control.throttle = self.power_level
        vehicle_control.steer = self.steer_angle / 40
        self.__vehicle.apply_control(vehicle_control)

class CarlaEgoVehicle(EgoVehicle):
    _vehicle: any
    _session: CarlaSession
    _state: VehicleState
    _gps: CarlaGPS
    _imu: CarlaIMU
    _odometer: CarlaOdometer
    _bev_semantic: BevCameraSemantic
    
    def __init__(self, session: CarlaSession, vehicle: any):
        super().__init__()
        self._session = session
        self._vehicle = vehicle
        self._state = None
        self._odometer = CarlaOdometer(vehicle)
        self.__clear_state()

    def __clear_state(self) -> None:
        self._state = VehicleState(self._vehicle)
        self._state.apply()

    def set_pose(self, pose: MapPose) -> None:
        self.__clear_state()
        x = pose.x
        y = pose.y
        z = pose.z
        yaw = pose.heading.rad()
        t = carla.libcarla.Transform(carla.libcarla.Location(x, y, z), carla.libcarla.Rotation(0, yaw, 0))
        self._vehicle.set_transform(t)

    def set_carla_autopilot(self, value: bool) -> None:
        # Fallback of the control to the simulator for auto driving. This method should be used for debugging only
        # Args:
        #    value (bool): turns on/off the simulator autopilot
        self._vehicle.set_autopilot(value)

    def set_power(self, power_level: float) -> None:
        self._state.power_level = abs(power_level)
        self._state.reversed = power_level < 0
        self._state.brake_level = 0.0
        self._state.apply()
        
    def set_brake(self, brake_level: float) -> None:
        self._state.power_level = 0.0
        self._state.brake_level = brake_level
        self._state.apply()

    def set_steering(self, angle: float) -> None:
        self._state.steer_angle = angle
        self._state.apply()
        
    def attach_gps_sensor(self, period_ms: int) -> None:
        self._gps = CarlaGPS(self._session, self._vehicle, period_ms)
        return self._gps

    def attach_imu_sensor(self, period_ms: int) -> None:
        self._imu = CarlaIMU(self._session, self._vehicle, period_ms)
        return self._imu
    
    def get_odometer_sensor(self) -> CarlaOdometer:
        return self._odometer

    def get_accel_vehicle(self) -> tuple[float, float, float]:
        """
        Returns the acceleration of the vehicle in m/s².
        """
        if self._vehicle is None:
            return (0.0, 0.0, 0.0)
        accel = self._vehicle.get_acceleration()
        return (accel.x, accel.y, accel.z)
    
    def read_gps(self) -> GpsData:
        if self._gps is None:
            return None
        return self._gps.read()
    
    def read_imu(self) -> IMUData:
        if self._imu is None:
            return None
        return self._imu.read()
    
    def read_odometer(self) -> float:
        return self._odometer.read()

    def get_carla_obj(self) -> any:
        return self._vehicle
    
    def init_semantic_bev_camera(self, width: int = 256, height: int = 256) -> BevCameraSemantic:
        self._bev_semantic = BevCameraSemantic(self._session, self._vehicle, width, height)
        return self._bev_semantic
    
    def destroy(self) -> None:
        self._vehicle.destroy()