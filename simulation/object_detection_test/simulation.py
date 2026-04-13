from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import EgoVehicle, DiscreteComponent, Camera
import time, cv2, numpy as np
from pydriveless import SearchFrame
from ultralytics import YOLO

SEGMENTED_COLORS = np.array([
        [0,   0,   0],
        [128,  64, 128],
        [244,  35, 232],
        [70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [0,   0, 142],
        [0,   0,  70],
        [0,  60, 100],
        [0,  80, 100],
        [0,   0, 230],
        [119,  11,  32],
        [110, 190, 160],
        [170, 120,  50],
        [55,  90,  80],     # other
        [45,  60, 150],
        [157, 234,  50],
        [81,   0,  81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180]
    ])
SEGMENTATION_CLASS_COST = np.array([
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        0,
        0,   # LAMP? investigate...
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1, # car
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        -1
    ], dtype=np.float32)


class ObjectPredictor(DiscreteComponent):
    _cameras: list[Camera]
    _front_frame: SearchFrame
    
    def __init__(self, period_ms, vehicle: CarlaEgoVehicle, img_size: int = 256):
        super().__init__(period_ms)
        self._vehicle = vehicle
        self._cameras = [
            self._vehicle.init_semantic_front_camera(width=img_size, height=img_size),
            self._vehicle.init_semantic_left_camera(width=img_size, height=img_size),
            self._vehicle.init_semantic_right_camera(width=img_size, height=img_size),
            self._vehicle.init_semantic_back_camera(width=img_size, height=img_size)
        ]
        self._all_images = np.zeros((3*img_size, 3*img_size, 3), dtype=np.uint8)
        self._img_size = img_size

        self._front_frame = SearchFrame(self._img_size, self._img_size, lower_bound=(-1, -1), upper_bound=(-1, -1))
        self._front_frame.set_class_colors(SEGMENTED_COLORS)
        self._front_frame.set_class_costs(SEGMENTATION_CLASS_COST)

        self._left_frame = SearchFrame(self._img_size, self._img_size, lower_bound=(-1, -1), upper_bound=(-1, -1))
        self._left_frame.set_class_colors(SEGMENTED_COLORS)
        self._left_frame.set_class_costs(SEGMENTATION_CLASS_COST)

        self._right_frame = SearchFrame(self._img_size, self._img_size, lower_bound=(-1, -1), upper_bound=(-1, -1))
        self._right_frame.set_class_colors(SEGMENTED_COLORS)
        self._right_frame.set_class_costs(SEGMENTATION_CLASS_COST)

        self._back_frame = SearchFrame(self._img_size, self._img_size, lower_bound=(-1, -1), upper_bound=(-1, -1))
        self._back_frame.set_class_colors(SEGMENTED_COLORS)
        self._back_frame.set_class_costs(SEGMENTATION_CLASS_COST)        

    def _copy_img(self, img: np.ndarray, x_start: int, z_start: int):
        self._all_images[
                int(z_start):int(z_start + self._img_size),
                int(x_start):int(x_start + self._img_size)
        ] = img


    def _loop(self, dt: float) -> None:
        cam_f, dt = self._cameras[0].read()
        cam_l, dt = self._cameras[1].read()
        cam_r, dt = self._cameras[2].read()
        cam_b, dt = self._cameras[3].read()
        
        if cam_f is not None:
            self._front_frame.set_frame_data(cam_f)
            self._copy_img(self._front_frame.get_color_frame(), self._img_size, 0)
        if cam_l is not None:
            self._left_frame.set_frame_data(cam_l)
            self._copy_img(self._left_frame.get_color_frame(), 0, self._img_size)           
        if cam_r is not None:
            self._right_frame.set_frame_data(cam_r)           
            self._copy_img(self._right_frame.get_color_frame(), 2*self._img_size, self._img_size)
        if cam_b is not None:
            self._back_frame.set_frame_data(cam_b)           
            self._copy_img(self._back_frame.get_color_frame(), self._img_size, 2*self._img_size)

        cv2.imwrite("teste.png", self._all_images)
        pass

def main():
    sim = CarlaSimulation(town_name="Town07")
    vehicle = sim.add_ego_vehicle(pos=[-90.0, 0, 2.0])
    #time.sleep(2)
    
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



    