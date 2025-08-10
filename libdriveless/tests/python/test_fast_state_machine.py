import sys, time
import unittest, math, numpy as np
import matplotlib.pyplot as plt
from pydriveless import FastStateMachine
from enum import Enum


class SingleStateTestMachine (FastStateMachine):
    i: int

    def __init__(self, debug = False, initial_state = 0):
        self.i = 0
        super().__init__({0 : "inc_i"}, debug, initial_state)

    # locked inside this state, forever incrementing i    
    def inc_i(self, state: int) -> int:
        self.i += 1
        return state

class DualStateTestMachine (FastStateMachine):
    values: list[int]

    def __init__(self, debug = False, initial_state = 0):
        self.values = []
        
        super().__init__({
            0 : "inc_i",
            1 : "inc_j",
        }, debug, initial_state)

    def inc_i(self, state: int) -> int:
        self.values.append(1)
        return 1

    def inc_j(self, state: int) -> int:
        self.values.append(2)
        return 0

class DualEnumState(Enum):
    STATE_1 = 0
    STATE_2 = 1

class DualEnumStateTestMachine (FastStateMachine):
    values: list[int]

    def __init__(self, debug = False, initial_state = DualEnumState.STATE_1):
        self.values = []
        
        super().__init__({
            DualEnumState.STATE_1 : "inc_i",
            DualEnumState.STATE_2 : "inc_j",
        }, debug, initial_state)

    def inc_i(self, state: DualEnumState) -> DualEnumState:
        self.values.append(1)
        return DualEnumState.STATE_2

    def inc_j(self, state: DualEnumState) -> DualEnumState:
        self.values.append(2)
        return DualEnumState.STATE_1


class TestFastStateMachine(unittest.TestCase):

  
    def test_no_state_running(self):
        sm = FastStateMachine(state_map={
            0 : None
        })
        sm.start()
        self.assertTrue(sm.is_running())
        time.sleep(0.3)
        sm.destroy()
 

    def test_single_state(self):
        sm = SingleStateTestMachine(debug=False)
        sm.start()
        self.assertTrue(sm.is_running())
        time.sleep(0.3)
        self.assertTrue(sm.i > 0)
        sm.destroy()

    def test_dual_state(self):
        sm = DualStateTestMachine(debug=False)
        sm.start()
        self.assertTrue(sm.is_running())
        time.sleep(0.3)
        
        self.assertTrue(len(sm.values) > 0)
        for i in range(len(sm.values)):
            if i % 2 == 0:
                self.assertEqual(1, sm.values[i])
            else:
                self.assertEqual(2, sm.values[i])
        sm.destroy()

    def test_dual_enum_state(self):
        sm = DualEnumStateTestMachine(debug=False)
        sm.start()
        self.assertTrue(sm.is_running())
        time.sleep(0.3)
        
        self.assertTrue(len(sm.values) > 0)
        for i in range(len(sm.values)):
            if i % 2 == 0:
                self.assertEqual(1, sm.values[i])
            else:
                self.assertEqual(2, sm.values[i])
        sm.destroy()

if __name__ == "__main__":
    unittest.main()
