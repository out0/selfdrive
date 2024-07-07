#
# Testing the datalink using separated process
#
import sys
sys.path.append("..")
from pybind.pydatalink import PyDatalink
import time
import numpy as np

def main ():
    link = PyDatalink(port=21001, timeout_ms=500)

    print ("waiting for client to connect")
    while not link.is_connected():
        time.sleep(0.01)

    pos = 1
    while True:
        if not link.is_connected():
            time.sleep(0.01)
            continue
        
        print (f"sending {pos}")
        link.write(f"data #{pos}")
        pos += 1
        time.sleep(0.5)


if __name__ == "__main__":
    main()