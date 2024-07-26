import sys, time
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
from motion.lateral_controller import LateralController
from model.map_pose import MapPose
from slam.slam import SLAM

class TestLateralController(unittest.TestCase):

    class StubSlamTest(SLAM):
        _pose: MapPose
        _last_steer_value: float

        def __init__(self) -> None:
            self._pose = None

        def set_pose(self, pose: MapPose):
            self._pose = pose
        
        def estimate_ego_pose(self) -> MapPose:
            return self._pose

    def steer_actuator(self, value: float) -> None:
        self._last_steer_value = value
        

    def test_straight(self):
        slam = TestLateralController.StubSlamTest()
        controller = LateralController(
            vehicle_length=4,
            slam=slam,
            velocity_read=lambda : 10,
            steering_actuator=self.steer_actuator
        )

        slam.set_pose(MapPose(0, 0, 0, 0))

        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertAlmostEqual(first=self._last_steer_value, second=0, msg="steering should be 0")

    def test_on_path_wrong_heading(self):
        slam = TestLateralController.StubSlamTest()
        controller = LateralController(
            vehicle_length=4,
            slam=slam,
            velocity_read=lambda : 10,
            steering_actuator=self.steer_actuator
        )

        slam.set_pose(MapPose(0, 0, 30, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertEqual(first=self._last_steer_value, second=-40, msg="steering should be -40")

        slam.set_pose(MapPose(0, 0, -30, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertEqual(first=self._last_steer_value, second=40, msg="steering should be +40")

    def test_right_heading_crosstrack_error(self):
        slam = TestLateralController.StubSlamTest()
        controller = LateralController(
            vehicle_length=4,
            slam=slam,
            velocity_read=lambda : 30,
            steering_actuator=self.steer_actuator
        )

        slam.set_pose(MapPose(0, 2, 0, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertTrue(self._last_steer_value < -10)

        slam.set_pose(MapPose(0, -2, 0, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertTrue(self._last_steer_value > 10)

    def test_right_heading_crosstrack_error_high_speed(self):
        slam = TestLateralController.StubSlamTest()
        controller = LateralController(
            vehicle_length=4,
            slam=slam,
            velocity_read=lambda : 100,
            steering_actuator=self.steer_actuator
        )

        slam.set_pose(MapPose(0, 2, 0, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertTrue(self._last_steer_value < -2)

        slam.set_pose(MapPose(0, -2, 0, 0))
        controller.set_reference_path(MapPose(-10, 0, 0, 0), MapPose(10, 0, 0, 0))
        controller.loop(time.time)
        self.assertTrue(self._last_steer_value > 2)

if __name__ == "__main__":
    unittest.main()


