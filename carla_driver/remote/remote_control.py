#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydatalink import Datalink
import numpy as np
import time
#import faulthandler
#faulthandler.enable()

CMD_SIZE = 10
KEEP_ALIVE_RESPONSE = np.zeros(CMD_SIZE, dtype=np.float32)

def execute_cmd(control_link, cmd):
    cmd_type = cmd[0]
    match cmd_type:
        case 0.0:
            # keep alive
            control_link.write(KEEP_ALIVE_RESPONSE)
        case 1.0:
            throttle = cmd[1]
            steer = cmd[2]
            brake = cmd[3]
            print (f"received: DRIVE command - throttle: {throttle}, steer: {steer}, brake: {brake}")
        case 2.0:
            pass
        case _:
            print (f"received: UNKNOWN command type: {cmd_type}")

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

    while True:
        if not control_link.is_ready():
            print ("waiting for the control client to login")
            while not control_link.is_ready():
                pass
            print ("ready to receive commands")
        
        cmd, size = control_link.read_np(shape=(CMD_SIZE), dtype=np.float32)
        if size == 0: continue

        execute_cmd(control_link, cmd)

        



if __name__ == "__main__":
    main()