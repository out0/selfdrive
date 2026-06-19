from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import EgoVehicle, DiscreteComponent, Camera
import time, cv2, numpy as np
from pydriveless import SearchFrame, MapPose

def main():
    sim = CarlaSimulation(town_name="Town07")
    
    path = [
        MapPose(-70.0, 0, 0), MapPose(-30.0, 0, 0),
        MapPose(-2.0, -12.0, 0), MapPose(-22.0, -64.0, 0),
        MapPose(-100.0, -64.0, 0),
        MapPose(-87.0, -114.0, 0), MapPose(-77.0, -144.0, 0),
        MapPose(-57.0, -157.0, 0), MapPose(-2.0, -157.0, 0),
        MapPose(-2.0, -97.0, 0),
        MapPose(-2.0, -12.0, 0),
        MapPose(-30.0, 0, 0),
        MapPose(-70.0, 0, 0)
    ]
    
    
    
    sim.show_path(path)
    
    
    #vehicle = sim.add_ego_vehicle(pos=[-90.0, 0, 2.0])
    #time.sleep(2)
    
    input("enter to finish")
    sim.clear_last_path()
    #vehicle.set_carla_autopilot(False)
    #time.sleep(1)
    #vehicle.destroy()
       

if __name__ == "__main__":
    main()



    