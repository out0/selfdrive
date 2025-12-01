#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydatalink import Datalink
import numpy as np
#import faulthandler
#faulthandler.enable()

def main():
    # print ("connecting to the simulator...")
    # sim = CarlaSimulation(
    #     town_name='Town07'
    # )
    # print ("summoning the EGO vehicle...")
    # ego = sim.add_ego_vehicle(
    #     pos=(-90.0, 0, 0), 
    #     rotation=(0, 0, 0))
    
    print ("setting up the control link...")
    control_link = Datalink(port=21001, timeout=2000)

    i = 0
    while True:
        if not control_link.is_ready():
            print ("waiting for the control client to login")
            while not control_link.is_ready():
                pass
            print ("ready to receive commands")
        
        #cmd, size = control_link.read_np(shape=(10), dtype=np.float32)
        #cmd, size = control_link.read_bytes()
        cmd, size = control_link.read_np(shape=(1024*1024), dtype=np.int8)
        if size == 0: continue

        print (f"{i} received cmd code: {cmd[0]}")
        i += 1




if __name__ == "__main__":
    main()