from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import EgoVehicle, DiscreteComponent, Camera
import time, cv2

def initialize() -> tuple[CarlaSimulation, CarlaEgoVehicle]:
    sim = CarlaSimulation(town_name="Town07")
    return sim, sim.add_ego_vehicle(pos=[-90.0, 0, 2.0])

class ObjectPredictor(DiscreteComponent):
    def __init__(self, period_ms, camera: Camera):
        super().__init__(period_ms)
        self._camera = camera
        
        
    def _loop(self, dt: float) -> None:
        data, timestamp = self._camera.read()
        if data is None:
            return
        cv2.imwrite("teste.png", data)
        pass

    

def main():
    sim, vehicle = initialize()
    cam = vehicle.init_rgb_front_camera(width=512, height=512)
    
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



    