#! /usr/bin/python3
#from carladriver import CarlaEgoVehicle, CarlaSimulation
from pydatalink import Datalink
import numpy as np
import time
import threading

CMD_SIZE = 10
KEEP_ALIVE_RESPONSE = np.zeros(CMD_SIZE, dtype=np.float32)

def execute_client_cmd(control_link, cmd):
    cmd_type = cmd[0]
    match cmd_type:
        case 0.0:
            # keep alive response
            print ("received: KEEP ALIVE response")
        case _:
            print (f"received: UNKNOWN response type: {cmd_type}")

def keep_alive_thr(control_link):
    while True:
        cmd = np.zeros(CMD_SIZE, dtype=np.float32)
        cmd[0] = 0.0
        control_link.write(cmd)
        time.sleep(0.5)

def main():
    
    control_link = Datalink(host="127.0.0.1", port=21001, timeout=2000)
    keep_alive_thread = threading.Thread(target=keep_alive_thr, args=(control_link,))
    keep_alive_thread.daemon = True
    keep_alive_thread.start()

    i = 0
    while True:
        if not control_link.is_ready():
            print ("connecting...")
            while not control_link.is_ready():
                pass

        print ("sending control cmd")    
        cmd = np.zeros(CMD_SIZE, dtype=np.float32)
        cmd[0] = 2.0
        control_link.write(cmd)
        i += 1

        if control_link.has_data():
            resp, size = control_link.read_np(shape=(10), dtype=np.float32)
            execute_client_cmd(control_link, resp)
               
        time.sleep(1)


if __name__ == "__main__":
    main()