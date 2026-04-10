from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import EgoVehicle, DiscreteComponent, Camera
import time, cv2
from ultralytics import YOLO

class ObjectPredictor(DiscreteComponent):
    def __init__(self, period_ms, camera: Camera):
        super().__init__(period_ms)
        self._camera = camera
        
        
    def _loop(self, dt: float) -> None:
        data, timestamp = self._camera.read()
        
        model = YOLO("yolov8n.pt")

        # Run prediction on your PNG file
        results = model(data)  # replace with your file path

        # Save results to disk
        results[0].save(filename="output.png")
        
        if data is None:
            return
        cv2.imwrite("teste.png", data)
        pass

def initialize() -> tuple[CarlaSimulation, CarlaEgoVehicle]:
    sim = CarlaSimulation(town_name="Town07")
    return sim, sim.add_ego_vehicle(pos=[-90.0, 0, 2.0])    

def main():
    sim, vehicle = initialize()
    img_size = 1024
    cam = vehicle.init_rgb_front_camera(width=img_size, height=img_size)
    
    predictor = ObjectPredictor(100, cam)
    predictor.start()
    vehicle.set_carla_autopilot(True)
    input("enter to finish")
    vehicle.set_carla_autopilot(False)
    time.sleep(1)
    predictor.destroy()
    vehicle.destroy()
    
    
    


if __name__ == "__main__":
    main()



    