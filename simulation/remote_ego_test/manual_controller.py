#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydriveless import RemoteEgoClient
import time

def main():
    ego = RemoteEgoClient()

    while True:
        for i in range(0, 40):
            ego.set_steering(i)
            time.sleep(0.1)
        
        for i in range(0, 40):
            ego.set_steering(40 - i)
            time.sleep(0.1)
        
        for i in range(0, 40):
            ego.set_steering(-i)
            time.sleep(0.1)
        
        for i in range(0, 40):
            ego.set_steering(i-40)
            time.sleep(0.1)


if __name__ == "__main__":
    main()