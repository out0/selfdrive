#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydatalink import Datalink
import numpy as np
import time


def main():
    
    control_link = Datalink(host="127.0.0.1", port=21001, timeout=2000)

    i = 0
    while True:
        if not control_link.is_ready():
            print ("connecting...")
            while not control_link.is_ready():
                pass
            
        cmd = np.zeros(1024*1024, dtype=np.int8)
        print (f"{i} data sending: {cmd.shape}")
        cmd[0] = 2.0
        control_link.write(cmd)
        i += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()