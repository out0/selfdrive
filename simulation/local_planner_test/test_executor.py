from test_utils import *
from pydriveless import EgoParams, SearchParams
from pyfastrrt import FastRRT




def exec_fastrrt_gpu(conf: TestConfig) -> FastRRT:
    ego_params: EgoParams = TestUtils.build_ego_params(conf)
    search_params: SearchParams = TestUtils.build_search_params(conf, gpu=True)

    # search_params.frame.process_safe_distance_zone((20, 20), compute_vectorized=True)
    # search_params.frame.process_distance_to_goal(search_params.goal.x, search_params.goal.z)

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

    path = planner.get_planned_path(False)
    with open("path.txt", "w") as arq:
         for p in path:
              arq.write(f"{p}\n")

    while planner.path_optimize():
         pass

    if planner.goal_reached():
        path = planner.get_planned_path()
        print(f"FastRRT (GPU) found a solution with {len(path)} nodes")

        path = planner.get_planned_path(True)
        TestUtils.export_planner_result(conf, "output.png", path)
        
    
    

    # )


if __name__ == "__main__":
    conf = TestUtils.read_config("map_cost_18")
    exec_fastrrt_gpu(conf)
    pass