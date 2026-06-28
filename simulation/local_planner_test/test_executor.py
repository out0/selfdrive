from pydriveless import EgoParams, SearchParams, TestUtils, TestConfig, TestTimer, Waypoint
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
    
def exec_fastrrt_gpu(conf: TestConfig) -> FastRRT:
    ego_params: EgoParams = TestUtils.build_ego_params(conf)
    search_params: SearchParams = TestUtils.build_search_params(conf, gpu=True)

    search_params.frame.process_safe_distance_zone((20, 20), compute_vectorized=True)
    search_params.frame.process_distance_to_goal(search_params.goal.x, search_params.goal.z)

    planner = FastRRT(ego_params=ego_params)
    planner.set_plan_data(search_params)

    TestTimer.exec_start("fastrrt_gpu")
    planner.search_init()
    search_params.frame.process_safe_distance_zone((20,20), compute_vectorized=True)

#    TestUtils.export_safe_distance_frame(search_params.frame, "output.png")

    num_loops = 0
    while planner.loop(smart=True):
        nodes = planner.export_graph_nodes()
        f = TestUtils.export_color_frame(conf, None)
        for i in range(nodes.shape[0]):
                x = nodes[i, 0]
                z = nodes[i, 1]
                f[z, x, :] = [0, 0, 255]
        cv2.imwrite("output.png", f)
        num_loops += 1

    nodes = planner.export_graph_nodes()
    f = TestUtils.export_color_frame(conf, None)
    for i in range(nodes.shape[0]):
        x = nodes[i, 0]
        z = nodes[i, 1]
        f[z, x, :] = [0, 0, 255]
    cv2.imwrite("output.png", f)

    TestTimer.exec_stop("fastrrt_gpu")

    path, _ = planner.get_planned_path(False)
    with open("path.txt", "w") as arq:
         for p in path:
              arq.write(f"{p}\n")

    while planner.path_optimize():
         pass

    if planner.goal_reached():
        path, _ = planner.get_planned_path()
        print(f"FastRRT (GPU) found a solution with {len(path)} nodes")

        path, _ = planner.get_planned_path(True)
        TestUtils.export_planner_result(conf, "output.png", path)


def exec_fastrrt_gpu_path_improv(conf: TestConfig, max_timeout_improving: int) -> FastRRT:
    ego_params: EgoParams = TestUtils.build_ego_params(conf)
    search_params: SearchParams = TestUtils.build_search_params(conf, gpu=True)

    search_params.frame.process_safe_distance_zone((20, 20), compute_vectorized=True)
    search_params.frame.process_distance_to_goal(search_params.goal.x, search_params.goal.z)

    planner = FastRRT(ego_params=ego_params)
    planner.set_plan_data(search_params)

    TestTimer.exec_start("fastrrt_gpu")
    planner.search_init()
    search_params.frame.process_safe_distance_zone((20,20), compute_vectorized=True)

#    TestUtils.export_safe_distance_frame(search_params.frame, "output.png")

    # num_loops = 0
    t = time.time()
    best_path = PathResult()
    while (time.time() - t) < max_timeout_improving:
        if not planner.loop(smart=True):                
            path, cost = planner.get_planned_path()
            best_path.replace_if_better(path, cost)

    TestTimer.exec_stop("fastrrt_gpu")

    while planner.path_optimize():
         pass

    if planner.goal_reached():
        path = planner.get_planned_path()
        print(f"FastRRT (GPU) found a solution with {len(path)} nodes")

        path, cost = planner.get_planned_path(True)
        if cost > 0:
            TestUtils.export_planner_result(conf, "output.png", path)
   



if __name__ == "__main__":
    conf = TestUtils.read_config("map_cost_8")
    exec_fastrrt_gpu_path_improv(conf, 0.5)
    pass