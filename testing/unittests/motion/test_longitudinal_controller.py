import sys, time
sys.path.append("../../")
sys.path.append("../../../")
import unittest
from motion.longitudinal_controller import LongitudinalController

class TestLongitudinalController(unittest.TestCase):
    _power_actuator_value: float
    _brake_actuator_value: float

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._power_actuator_value = 0
        self._brake_actuator_value = 0
    
    
    def power_actuator(self, value: float) -> None:
        self._power_actuator_value = value
    
    def brake_actuator(self, value: float) -> None:
        self._brake_actuator_value = value

    def test_accel(self):  
        curr_speed = 0
        controller = LongitudinalController(
            actuation_period_ms=20,
            power_actuator=self.power_actuator,
            brake_actuator=self.brake_actuator,
            velocity_read=lambda : curr_speed
        )

        controller.set_speed(20.0)
        controller.loop(0.001)

        self.assertEqual(0.1, self._power_actuator_value)
        self.assertEqual(0.0, self._brake_actuator_value)

    def test_slight_deceleration(self):  
        curr_speed = 20
        controller = LongitudinalController(
            actuation_period_ms=20,
            power_actuator=self.power_actuator,
            brake_actuator=self.brake_actuator,
            velocity_read=lambda : curr_speed
        )

        controller.set_speed(0.0)
        controller.loop(0.001)

        self.assertEqual(0.0, self._power_actuator_value)
        self.assertEqual(0.3, self._brake_actuator_value)

    def test_moderate_deceleration(self):  
        curr_speed = 40
        controller = LongitudinalController(
            actuation_period_ms=20,
            power_actuator=self.power_actuator,
            brake_actuator=self.brake_actuator,
            velocity_read=lambda : curr_speed
        )

        controller.set_speed(0.0)
        controller.loop(0.001)

        self.assertEqual(0.0, self._power_actuator_value)
        self.assertEqual(0.5, self._brake_actuator_value)

    def test_strong_deceleration(self):  
        curr_speed = 100
        controller = LongitudinalController(
            actuation_period_ms=20,
            power_actuator=self.power_actuator,
            brake_actuator=self.brake_actuator,
            velocity_read=lambda : curr_speed
        )

        controller.set_speed(0.0)
        controller.loop(0.001)

        self.assertEqual(0.0, self._power_actuator_value)
        self.assertEqual(1.0, self._brake_actuator_value)

if __name__ == "__main__":
    unittest.main()


