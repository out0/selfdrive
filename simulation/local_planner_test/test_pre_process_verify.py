from test_utils import *
from pydriveless import EgoParams, SearchParams, SearchFrame
from pyfastrrt import FastRRT


if __name__ == "__main__":
    conf = TestUtils.read_config("map_cost_5")
    TestUtils.export_color_frame(conf, "output.png")
    frame: SearchFrame = TestUtils.build_cuda_frame(conf)
    frame.process_safe_distance_zone((10, 10), True)
    frame.process_distance_to_goal(428, 338)
    TestUtils.export_safe_distance_frame_minimal_dist_flag(frame, "output2.png")
    pass