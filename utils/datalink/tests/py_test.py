import sys
sys.path.append("..")
import unittest
from pybind import pydatalink
import time
import numpy as np

server = pydatalink.PyDatalink(port=20000, timeout_ms=500)
client = pydatalink.PyDatalink(host= "127.0.0.1", port=20000, timeout_ms=500)

class TestDataLink(unittest.TestCase):
    _start_time: int = 0

    def rst_timeout(self):
        self._start_time = -1

    def timeout_check(self, timeout_s: float):
        if self._start_time <= 0:
            self._start_time = time.time()
            return False
        return time.time() - self._start_time > timeout_s


    def test_transmit_data(self):

        while not client.is_connected():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(1)

        self.rst_timeout()
        while not server.is_connected():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(1)

        self.assertTrue(server.is_connected())
        self.assertTrue(client.is_connected())


        message = "Message1"
        self.assertTrue(client.write(message))
        self.rst_timeout()

        while not server.has_data():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)

        received_message, size = server.read()
        self.assertEqual(message, received_message)

    def test_transmit_bidirectional_data(self):
        
        self.rst_timeout()
        while not client.is_connected():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)


        self.assertTrue(server.is_connected())
        self.assertTrue(client.is_connected())


        message = "Message1"

        self.assertTrue(client.write(message))
        self.rst_timeout()        
        while not server.has_data():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)

        received_message, size = server.read()
        self.assertEqual(message, received_message)

        self.assertTrue(server.write(message))
        self.rst_timeout()        
        while not client.has_data():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)

        received_message, size = client.read()
        self.assertEqual(message, received_message)

    def test_echo_np_array(self):
        data = np.full((401, 400), 23.23, dtype=np.float32)
        
        self.rst_timeout()
        while not client.is_connected():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)

        
        server.write_float_np(data)
        
        self.rst_timeout()
        while not client.has_data():
            if self.timeout_check(2):
                self.fail("timeout after waiting for 2s")
            time.sleep(0.01)

        data2 = client.read_float_np((401, 400))

        for i in range (data.shape[0]):
            for j in range (data.shape[1]):
                if (data[i, j] != data2[i, j]):
                    self.fail("the received np.array is corrupt at ({i}, {j}): sent {data[i,j]} <> received {data2[i,j]}")
                

    def test_send_multiple_np_array_parallel(self):
        data = np.full((401, 400), 23.23, dtype=np.float32)
        pair_count:int = 10

        server_links: list[pydatalink.PyDatalink] = []
        client_links: list[pydatalink.PyDatalink] = []

        print(f"creating {pair_count} pair connections")
        for i in range(pair_count):
            server_links.append(pydatalink.PyDatalink(port=21000 + i, timeout_ms=500))
            client_links.append(pydatalink.PyDatalink(host="127.0.0.1", port=21000 + i, timeout_ms=500))
        
        print(f"waiting for {pair_count} pair connections to be connected")
        for i in range(pair_count):
            self.assertTrue(server_links[i].wait_ready())
            self.assertTrue(client_links[i].wait_ready())

        for i in range(pair_count):
            print(f"sending array to #{i} pair connection")
            self.assertTrue(server_links[i].write_float_np(data))

        rcv_data: list[np.ndarray] = []

        for i in range(pair_count):
            print(f"receiving array from #{i} pair connection")
            res = client_links[i].read_float_np((401, 400))
            self.assertIsNotNone(res)
            rcv_data.append(res)

        print(f"checking data for {pair_count} responses")
        for p in range(pair_count):
            for i in range (data.shape[0]):
                for j in range (data.shape[1]):
                    if (data[i, j] != rcv_data[p][i, j]):
                        self.fail("the received np.array is corrupt at ({i}, {j}): sent {data[i,j]} <> received {data2[i,j]}")
        
        for i in range(pair_count):
            server_links[i].terminate()
            client_links[i].terminate()

if __name__ == "__main__":
    unittest.main()