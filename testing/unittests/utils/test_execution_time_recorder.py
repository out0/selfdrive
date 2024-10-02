import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
import numpy as np
from utils.time_recorder import ExecutionTimeRecorder
import time
import os
from threading import Thread

class TestExecutionRecorder(unittest.TestCase): 
    def test_generate_write_data_1_thr(self):

        filename = ExecutionTimeRecorder.get_file_name()
        if os.path.exists(filename):
            os.remove(filename)

        time_start = time.time()
        ExecutionTimeRecorder.start('module1')
        time.sleep(0.1) # 100 ms
        ExecutionTimeRecorder.stop('module1')
        time_total = 1000 * (time.time() - time_start)
        
        self.assertTrue(time_total <= 105)  # assert it takes at most 5 ms to measure timing

        time.sleep(0.01) # 10 ms
        with open(filename, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)

            line1 = lines[0].replace("\n", "").split(':')
            self.assertEqual(line1[0], 'module1')
            exec_timing = float(line1[1])
            self.assertTrue(exec_timing < 102)

    def measure_item(self, i: int) -> None:
        ExecutionTimeRecorder.start(f'module{i}')
        time.sleep(1) # 100 ms
        ExecutionTimeRecorder.stop(f'module{i}')

    def test_generate_write_data_100_thr(self):

        filename = ExecutionTimeRecorder.get_file_name()
        if os.path.exists(filename):
            os.remove(filename)

        for i in range (0,100):
            thr = Thread(target=lambda : self.measure_item(i))
            thr.start()

        time.sleep(2) # 10 ms
        with open(filename, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 100)

            # line1 = lines[0].replace("\n", "").split(':')
            # self.assertEqual(line1[0], 'module1')
            # exec_timing = float(line1[1])
            # self.assertTrue(exec_timing < 102)

if __name__ == "__main__":
    unittest.main()