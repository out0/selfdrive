from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import EgoVehicle, DiscreteComponent, Camera
import time, cv2, numpy as np
from ultralytics import YOLO

class ObjectPredictor(DiscreteComponent):
    _cameras: list[Camera]
    
    def __init__(self, period_ms, vehicle: CarlaEgoVehicle, img_size: int = 256):
        super().__init__(period_ms)
        self._vehicle = vehicle
        self._cameras = [
            self._vehicle.init_rgb_front_camera(width=img_size, height=img_size),
            self._vehicle.init_rgb_left_camera(width=img_size, height=img_size),
            self._vehicle.init_rgb_right_camera(width=img_size, height=img_size),
            self._vehicle.init_rgb_back_camera(width=img_size, height=img_size)
        ]
        self._all_images = np.zeros((2*img_size, 3*img_size, 3), dtype=np.uint8)
        self._img_size = img_size

    def _copy_img(self, img: np.ndarray, x_start: int, z_start: int):
        self._all_images[
                int(z_start):int(z_start + self._img_size),
                int(x_start):int(x_start + self._img_size)
        ] = img


    def _loop(self, dt: float) -> None:
        cam_f, dt = self._cameras[0].read()
        cam_l, dt = self._cameras[1].read()
        cam_r, dt = self._cameras[2].read()
        self._copy_img(cam_f, self._img_size, 0)
        self._copy_img(cam_l, 0, self._img_size)
        self._copy_img(cam_r, 2*self._img_size, self._img_size)
        cv2.imwrite("teste.png", self._all_images)
        pass

def main():
    sim = CarlaSimulation(town_name="Town07")
    vehicle = sim.add_ego_vehicle(pos=[-90.0, 0, 2.0])
    
    predictor = ObjectPredictor(100, vehicle)
    predictor.start()
    vehicle.set_carla_autopilot(True)
    input("enter to finish")
    vehicle.set_carla_autopilot(False)
    time.sleep(1)
    predictor.destroy()
    vehicle.destroy()
    
    
    


if __name__ == "__main__":
    main()



    