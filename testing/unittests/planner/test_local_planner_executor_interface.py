import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor

class DummyLocalPlanner (LocalPathPlannerExecutor):
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
    
    def check_timeout(self) -> bool:
        return self._check_timeout()


class TestLocalPlannerExecutorInterface(unittest.TestCase):
    
    def test_timeout_trigger(self):
      pl = DummyLocalPlanner(200)
      pl.set_exec_started()
      self.assertFalse(pl.check_timeout())
      time.sleep(0.3)
      self.assertTrue(pl.check_timeout())
      self.assertAlmostEqual(pl.get_execution_time() / 1000, 0.3, places=1)
      
    

if __name__ == "__main__":
    unittest.main()
