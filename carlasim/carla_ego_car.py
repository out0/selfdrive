from .carla_client import CarlaClient
import random
import carla
from carlasim.sensors.carla_camera import *
from carlasim.sensors.data_sensors import CarlaGps, CarlaIMU, CarlaOdometer
from model.ego_car import EgoCar
import numpy as np



GPS_PERIOD_ms = 50
IMU_PERIOD_ms = 50



class CarlaEgoCar(EgoCar):
    _client: CarlaClient
    _ego_car: any
    _vehicle_control: carla.VehicleControl
    _camera_bev: BEVRGBCamera
    _camera_front: FrontRGBCamera
    _camera_back: BackRGBCamera
    _camera_left: LeftRGBCamera
    _camera_right: RightRGBCamera
    _gps: CarlaGps
    _imu: CarlaIMU
    _odometer: CarlaOdometer

    def __init__(self, client: CarlaClient, color: str = '63, 183, 183') -> None:
        self._client = client
        bp = self._client.get_blueprint("vehicle.tesla.model3")
        bp.set_attribute('color', color)
        location = random.choice(
            self._client.get_world().get_map().get_spawn_points())
        self._ego_car = self._client.get_world().spawn_actor(bp, location)
        self._vehicle_control = carla.VehicleControl()

        self._gps = CarlaGps(client, self._ego_car, GPS_PERIOD_ms/1000)
        self._imu = CarlaIMU(client, self._ego_car, IMU_PERIOD_ms/1000)
        self._odometer = CarlaOdometer(self._ego_car)
        self.remote_controller = None
        self._autonomous_mode = False
        self._slam = None
        self._camera_bev = None
        self._camera_front = None
        self._camera_back = None
        self._camera_left = None
        self._camera_right = None

    def init_fake_bev_camera(self, width: int = 256, height: int = 256) -> None:
        self._camera_bev = BEVRGBCamera(width, height, 30)
        self._camera_bev.attach_to(self._client, self._ego_car)

    def init_fake_bev_seg_camera(self, width: int = 256, height: int = 256) -> None:
        self._camera_bev = BEVSemanticCamera(width, height, 30)
        self._camera_bev.attach_to(self._client, self._ego_car)

    def init_surrounding_cameras(self, width: int = 256, height: int = 256) -> None:
        self._camera_front = FrontRGBCamera(width, height, 110, 30)
        self._camera_back = BackRGBCamera(width, height, 110, 30)
        self._camera_left = LeftRGBCamera(width, height, 110, 30)
        self._camera_right = RightRGBCamera(width, height, 110, 30)
        self._camera_front.attach_to(self._client, self._ego_car)
        self._camera_back.attach_to(self._client, self._ego_car)
        self._camera_left.attach_to(self._client, self._ego_car)
        self._camera_right.attach_to(self._client, self._ego_car)
   
    def init_surrounding_seg_cameras(self, width: int = 256, height: int = 256) -> None:
        self._camera_front = FrontSemanticCamera(width, height, 110, 30)
        self._camera_back = BackSemanticCamera(width, height, 110, 30)
        self._camera_left = LeftSemanticCamera(width, height, 110, 30)
        self._camera_right = RightSemanticCamera(width, height, 110, 30)
        self._camera_front.attach_to(self._client, self._ego_car)
        self._camera_back.attach_to(self._client, self._ego_car)
        self._camera_left.attach_to(self._client, self._ego_car)
        self._camera_right.attach_to(self._client, self._ego_car)
      
    def attach_custom_front_camera(self, camera : CarlaCamera) -> None:
        self._camera_front = camera
        self._camera_front.attach_to(self._client, self._ego_car)
    
    def attach_custom_back_camera(self, camera : CarlaCamera) -> None:
        self._camera_back = camera
        self._camera_back.attach_to(self._client, self._ego_car)
      
    def attach_custom_left_camera(self, camera : CarlaCamera) -> None:
        self._camera_left = camera
        self._camera_left.attach_to(self._client, self._ego_car)
      
    def attach_custom_right_camera(self, camera : CarlaCamera) -> None:
        self._camera_right = camera
        self._camera_right.attach_to(self._client, self._ego_car)
                           
    def __destroy_if_not_none(self, p: any):
        if p is None:
            return
        p.destroy()
        
    def destroy(self) -> None:
        self.__destroy_if_not_none(self._gps)
        self.__destroy_if_not_none(self._imu)
        self.__destroy_if_not_none(self._odometer)
        self.__destroy_if_not_none(self._camera_front)
        self.__destroy_if_not_none(self._camera_back)
        self.__destroy_if_not_none(self._camera_left)
        self.__destroy_if_not_none(self._camera_right)
        self.__destroy_if_not_none(self._ego_car)

    def get_location(self) -> np.array:
        # """ Returns the vehicle's location on the simulator. This value should be used for debugging only
        #
        # Returns:
        #     _type_: _description_
        # """

        p = self._ego_car.get_location()
        return np.array([p.x, p.y, p.z])

    def get_heading(self) -> float:
        # """Returns the vehicle's heading on the simulator. This value should be used for debugging only
        # Returns:
        #     float: _description_
        # """
        t = self._ego_car.get_transform()
        return t.rotation.yaw

    def set_pose(self, x: int, y: int, z: int, heading: float) -> None:
        # """Sets a new pose for the vehicle by teleporting it to the target location
        # Args:
        #     x (int): x coordinate in carla's map
        #     y (int): y coordinate in carla's map
        #     z (int): z coordinate in carla's map
        #     heading (float): car's yaw representing it's heading
        # """
        t = self._ego_car.get_transform()
        t.location = carla.libcarla.Location(x, y, z)
        t.rotation.yaw = heading
        self._ego_car.set_transform(t)

    def set_autopilot(self, value: bool) -> None:
        # Fallback of the control to the simulator for auto driving. This method should be used for debugging only
        # Args:
        #    value (bool): turns on/off the simulator autopilot
        self._ego_car.set_autopilot(value)

    def set_power(self, power: float) -> None:
        # self._vehicle_control.throttle = abs(power / 240)

        #print(f"set_power({power})")

        self._vehicle_control.reverse = False
        self._vehicle_control.brake = 0.0
        self._vehicle_control.throttle = power
        if power < 0:
            self._vehicle_control.reverse = True
            self._vehicle_control.throttle = -power

        self._ego_car.apply_control(self._vehicle_control)

    def set_brake(self, brake: float) -> None:
        self._vehicle_control.throttle = 0.0
        self._vehicle_control.brake = brake
        self._ego_car.apply_control(self._vehicle_control)

    def set_steering(self, angle: int) -> None:
        self._vehicle_control.steer = angle / 40
        self._ego_car.apply_control(self._vehicle_control)

    def set_autonomous_mode_flag(self, val: bool) -> None:
        self._autonomous_mode = val

    def chk_autonomous_mode_flag(self) -> bool:
        return self._autonomous_mode

    def get_carla_ego_car_obj(self):
        return self._ego_car

    def get_odometer(self) -> CarlaOdometer:
        return self._odometer

    def get_gps(self) -> CarlaGps:
        return self._gps

    def get_imu(self) -> CarlaIMU:
        return self._imu

    def get_front_camera(self) -> CarlaCamera:
        return self._camera_front

    def get_left_camera(self) -> CarlaCamera:
        return self._camera_left

    def get_right_camera(self) -> CarlaCamera:
        return self._camera_right

    def get_back_camera(self) -> CarlaCamera:
        return self._camera_back
    
    def get_bev_camera(self) -> CarlaCamera:
        return self._camera_bev