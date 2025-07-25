#
# Testing the datalink using separated process
#
import sys
sys.path.append("..")
from pybind.pydatalink import PyDatalink
import time
import numpy as np
import cv2

def main ():
    link = PyDatalink(port=21001, timeout_ms=500)

    print ("waiting for client to connect")
    while not link.is_connected():
        time.sleep(0.01)

    #image = cv2.imread("./img_1.png")
    image = np.full((1024,1024,3), 11.2, np.float32)
    image = np.asarray(image)

    pos = 1
    while True:
        if not link.is_connected():
            time.sleep(0.01)
            continue
        
        print (f"sending {pos}")
        link.write_float_np(image)
        res, _ = link.read()
        if (res == "ack"):
            print("acked")
            #exit(0)


if __name__ == "__main__":
    main()