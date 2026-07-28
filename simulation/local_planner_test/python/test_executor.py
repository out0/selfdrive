from pydriveless import EgoParams, SearchParams, TestUtils, TestConfig, TestTimer, Waypoint
from pydriveless import LocalPlanner, SearchFrame
from pyfastrrt import FastRRT
import cv2, time

class PathResult:
    path: list[Waypoint]
    cost: float

    def __init__(self):
         self.path = None
         self.cost = -1
        
    def replace_if_better(self, path: list[Waypoint], cost: float):
        if cost < 0:
             return
        if self.path is None or self.cost > cost:
             if self.cost > 0:
                 print(f"replacing path. Old cost {self.cost} New cost: {cost}")
             
             self.path = path
             self.cost = cost


def export_fastrrt_nodes(planner: FastRRT):
    f = TestUtils.export_color_frame(conf, None)
    nodes = planner.export_graph_nodes()
    for i in range(nodes.shape[0]):
            x = nodes[i, 0]
            z = nodes[i, 1]
            f[z, x, :] = [0, 0, 255]
    cv2.imwrite("output.png", f)

def export_path_to_file(planner: FastRRT):
    path, _ = planner.get_planned_path(False)
    with open("path.txt", "w") as arq:
         for p in path:
              arq.write(f"{p}\n")


def exec_local_planner(planner: LocalPlanner, frame: SearchFrame, conf: TestConfig, max_timeout_improving: int) -> FastRRT:
    TestTimer.exec_start()
    planner.initialize(True)

    if max_timeout_improving > 0:
        # num_loops = 0
        t = time.time()
        best_path = PathResult()
        while (time.time() - t) < max_timeout_improving:
            if not planner.planning_loop():                
                path, cost = planner.get_planned_path()
                best_path.replace_if_better(path, cost)
    else:
         while planner.planning_loop():
              #TestUtils.export_frame_planner_result(conf, frame, "output2.png",)
              pass

    TestTimer.exec_stop()

    while planner.path_optimize_loop():
         pass

    if planner.goal_reached():
        path, cost = planner.get_planned_path()
        TestUtils.save_path(path, "path.txt")
        print(f"{planner.name()} found a solution with {len(path)} nodes")

        path, cost  = planner.get_interpolated_planned_path()
        TestUtils.export_frame_planner_result(conf, frame, "output.png", path)
   

def exec_fastrrt_costmap(map_name: str):
    conf = TestUtils.read_config(map_name)
    safe_dist = (10, 10)
    ego_params: EgoParams = TestUtils.build_ego_params(conf)
    search_params: SearchParams = TestUtils.build_search_params(conf, gpu=True)

    TestUtils.export_color_frame(conf, "output.png")

    search_params.frame.process_distance_to_goal(search_params.goal.x, search_params.goal.z)
    search_params.frame.process_safe_distance_zone(safe_dist, compute_vectorized=True)


    planner = FastRRT(ego_params=ego_params, smart=True)
    planner.set_plan_data(search_params)

    while True:
        exec_local_planner(planner, search_params.frame, conf, max_timeout_improving=-1)



if __name__ == "__main__":
    exec_fastrrt_costmap("map_cost_31")
    pass