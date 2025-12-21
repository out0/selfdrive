import sys, time
sys.path.append("../../")
sys.path.append("../")
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import SearchFrame, float3
from pydriveless import Telemetry, TelemetryReader
from pydatalink import Datalink
from test_utils import fix_cv2_import
import threading
from collections import deque
fix_cv2_import()


class TelemetryDataReader(TelemetryReader):
    rcv_data: list
    def __init__(self):
        super().__init__("127.0.0.1", 21001)
        self.rcv_data = []
    
    def _on_log_received (self, id: str, data: any, timestamp: float) -> None:
        self.rcv_data.append((id, data))


class TestTelemetry(unittest.TestCase):
        
    def test_mock_telemetry(self):
        Telemetry.setup(21001, 100)

        Telemetry.log(id="id1", data="data1")
        Telemetry.log(id="id2", data="data2")

        reader = TelemetryDataReader()
        #time.sleep(1.100)
        
        while len(reader.rcv_data) != 2:
            pass
        self.assertEqual(reader.rcv_data[0][0], "id1")
        self.assertEqual(reader.rcv_data[0][1], "data1")
        self.assertEqual(reader.rcv_data[1][0], "id2")
        self.assertEqual(reader.rcv_data[1][1], "data2")
        del reader
        Telemetry.terminate()




if __name__ == "__main__":
    unittest.main()
