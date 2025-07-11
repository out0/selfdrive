#
# Testing the datalink using separated process
#
import sys
sys.path.append("..")
from pybind.pydatalink import PyDatalink
import time
import numpy as np

def main ():
    link = PyDatalink(host="127.0.0.1", port=21001, timeout_ms=500)

    print ("connecting to the server")
    while not link.is_connected():
        time.sleep(0.01)

    while True:
        if link.has_data():
            arr = link.read_uint8_np((10, 10))
            print (arr.shape)

    # while True:
    #     if not link.is_connected():
    #         time.sleep(0.01)
    #         continue
        
    #     if link.has_data():
    #         msg, _ = link.read()
    #         print (f"received data: '{msg}'")
    #     time.sleep(0.01)


if __name__ == "__main__":
    main()