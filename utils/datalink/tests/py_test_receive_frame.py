#
# Testing the datalink using separated process
#
import sys
sys.path.append("..")
from pybind.pydatalink import PyDatalink
import time
import cv2

def main ():
    link = PyDatalink(host="127.0.0.1", port=21001, timeout_ms=500)

    print ("connecting to the server")
    while True:
        if not link.is_connected():
            time.sleep(0.01)
            continue

        if link.has_data():
            arr = link.read_float_np((1024, 1024, 3))
            link.write("ack")
        #cv2.imwrite("received_img.png", arr)
            print(f"received frame {arr.shape}")

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