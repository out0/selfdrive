from carladriver import CarlaSimulation
from pydriveless import MapPose

def main():
    # Initialize the Carla simulation
    sim = CarlaSimulation(town_name='Town07')
    
    path = [ MapPose(i, 0, 2) for i in range(1,30) ]
    path.extend([ MapPose(0, -i, 2) for i in range(1,30) ])
    
    sim.show_path(path)
    sim.show_coordinate((0, 0, 2))
    
    p = input("Press Enter to continue...")
    sim.reset()
    p = input("Press Enter to continue...(2)")
    

if __name__ == "__main__":
    main()