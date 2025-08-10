import sys, time
sys.path.append("../../")
import unittest

from pydriveless import DiscreteComponent

class TstDiscreteComp (DiscreteComponent):
    _last_loop_run: float

    def __init__(self, period_ms: int) -> None:
        super().__init__(period_ms)
        self._last_loop_run = 0

    def _loop(self, dt: float) -> None:
        self._last_loop_run = dt
        return super()._loop(dt)
    
    def get_last_loop_run(self) -> float:
        return self._last_loop_run

class TestDiscreteComponent(unittest.TestCase):

    def test_discrete_component_manual_loop(self):
        dc = TstDiscreteComp(10)
        dc.manual_loop_run(100)
        self.assertEqual(100, dc.get_last_loop_run())
        dc.destroy()

    def test_discrete_component_auto_loop(self):
        dc = TstDiscreteComp(10)
        dc.start()
        time.sleep(0.2)
        self.assertAlmostEqual(10/1000, dc.get_last_loop_run(), places=2)
        dc.destroy()


if __name__ == "__main__":
    unittest.main()
