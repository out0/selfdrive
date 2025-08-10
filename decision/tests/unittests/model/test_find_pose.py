import unittest
from pydriveless import MapPose
import json

class TestMapPose(unittest.TestCase):

    def test_find_nearest_goal_pose(self):
        return
        with open("motion_controller.log", "r") as f:
            lines = f.readlines()

            path = json.loads(lines[0])
            path = [MapPose.from_str(p) for p in path]

            investigate_line = lines[51]
            data = json.loads(investigate_line)
            pose = MapPose.from_str(data["pose"])
            pos = MapPose.find_nearest_goal_pose(pose, path, 0, max_hopping=len(path) - 1)

            # 
            a = 1

            # for i in range(1, len(lines)):
            #     data = json.loads(lines[i])
            #     pose = MapPose.from_str(data["pose"])
            #     pos = MapPose.find_nearest_goal_pose(pose, path, 0, maxHopping=len(path) - 1)
            #     self.assertEqual(pos, data["pos"])

        pass




if __name__ == "__main__":
    unittest.main()

