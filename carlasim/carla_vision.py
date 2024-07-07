from carlasim.carla_ego_car import EgoCar
import numpy as np
from model.vision import Vision

class CarlaVision (Vision):
    _ego_car: EgoCar

    def __init__(self, ego: EgoCar) -> None:
        super().__init__()
        self._ego_car = ego

    def read(self) -> np.array:
        return self._ego_car.bev_camera.read()