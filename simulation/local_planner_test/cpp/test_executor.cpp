// test_executor.cpp
//
// C++ equivalent of test_executor.py.
//
// Python originally used:
//   from pydriveless import EgoParams, SearchParams, TestUtils, TestConfig, TestTimer, Waypoint
//   from pydriveless import LocalPlanner, SearchFrame
//   from pyfastrrt import FastRRT
//
// Those python modules are thin ctypes wrappers around the real C++ classes
// living in libdriveless / FastRRT, so this file talks to that C++ API
// directly (driveless/*.h + FastRRT.h), and uses test_utils.h (in this same
// folder) as the C++ port of the python-only TestUtils/TestConfig/TestTimer
// test helpers.

#include <driveless/fastrrt.h>
#include <driveless/local_planner.h>
#include <driveless/search_frame.h>
#include <driveless/search_params.h>
#include <driveless/waypoint.h>
#include <driveless/interpolator.h>

#include "test_utils.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// -----------------------------------------------------------------------
// PathResult: mirrors the python PathResult helper class
// -----------------------------------------------------------------------
class PathResult
{
public:
    std::vector<Waypoint> path;
    float cost;
    bool has_path;

    PathResult() : cost(-1.0f), has_path(false) {}

    void replace_if_better(const std::vector<Waypoint> &new_path, float new_cost)
    {
        if (new_cost < 0)
            return;

        if (!has_path || cost > new_cost)
        {
            if (cost > 0)
                std::cout << "replacing path. Old cost " << cost << " New cost: " << new_cost << std::endl;

            path = new_path;
            cost = new_cost;
            has_path = true;
        }
    }
};

// -----------------------------------------------------------------------
// export_fastrrt_nodes: mirrors python's export_fastrrt_nodes()
// -----------------------------------------------------------------------
void export_fastrrt_nodes(FastRRT &planner, TestConfig &conf)
{
    cv::Mat f = TestUtils::export_color_frame(conf, "");
    std::vector<GraphNode> nodes = planner.exportGraphNodes();

    int width = f.cols;
    int height = f.rows;

    // Draw Hermite-interpolated curves from each node to its parent
    for (auto &n : nodes)
    {
        // Skip root nodes (parent == self)
        if (n.parent_x == -1 && n.parent_z == -1)
            continue;

        // Build waypoints with heading for both endpoints
        angle parentHeading = planner.getHeading(n.parent_x, n.parent_z);
        Waypoint wp_parent(n.parent_x, n.parent_z, parentHeading);
        Waypoint wp_child(n.x, n.z, angle::rad(n.heading_rad));

        // std::vector<Waypoint> curve = interpolateHermiteCurve(width, height, wp_parent, wp_child);

        std::vector<Waypoint> curve = Interpolator::hermite(width, height, wp_parent, wp_child, 0.1f);

        for (auto &pt : curve)
        {
            int px = pt.x();
            int pz = pt.z();
            if (px >= 0 && px < width && pz >= 0 && pz < height)
                f.at<cv::Vec3b>(pz, px) = cv::Vec3b(131, 143, 134);
        }
    }

    // Draw nodes on top so they remain visible
    for (auto &n : nodes)
        f.at<cv::Vec3b>(n.z, n.x) = cv::Vec3b(0, 0, 255);

    cv::imwrite("output.png", f);
}

// -----------------------------------------------------------------------
// export_path_to_file: mirrors python's export_path_to_file()
// -----------------------------------------------------------------------
void export_path_to_file(FastRRT &planner)
{
    auto [path, cost] = planner.getPlannedPath();
    (void)cost;

    std::ofstream arq("path.txt");
    for (auto &p : path)
        arq << "x=" << p.x() << ", z=" << p.z() << ", heading_deg=" << p.heading().deg() << "\n";
}

// -----------------------------------------------------------------------
// exec_local_planner: mirrors python's exec_local_planner()
// -----------------------------------------------------------------------
void exec_local_planner(FastRRT &planner, SearchFrame &frame, TestConfig &conf, int max_timeout_improving)
{
    TestTimer::exec_start();
    planner.initialize(true);

    if (max_timeout_improving > 0)
    {
        auto t = std::chrono::high_resolution_clock::now();
        PathResult best_path;

        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count() < max_timeout_improving)
        {
            if (!planner.planning_loop())
            {
                auto [path, cost] = planner.getPlannedPath();
                best_path.replace_if_better(path, cost);
            }
        }
    }
    else
    {
        while (planner.planning_loop())
        {
            //export_fastrrt_nodes(planner, conf);
        }
        auto [path, cost] = planner.getInterpolatedPlannedPath();
        TestUtils::export_frame_planner_result(conf, frame, "output2.png", path);
    }

    TestTimer::exec_stop();

    while (planner.path_optimize_loop())
    {
    }

    if (planner.goalReached())
    {
        auto [path, cost] = planner.getPlannedPath();
        (void)cost;
        TestUtils::save_path(path, "path.txt");
        std::cout << "FastRRT found a solution with " << path.size() << " nodes" << std::endl;

        auto [interp_path, interp_cost] = planner.getInterpolatedPlannedPath();
        (void)interp_cost;
        TestUtils::export_frame_planner_result(conf, frame, "output.png", interp_path);
    }
}

// -----------------------------------------------------------------------
// exec_fastrrt_costmap: mirrors python's exec_fastrrt_costmap()
// -----------------------------------------------------------------------
void exec_fastrrt_costmap(const std::string &map_name)
{
    TestConfig conf = TestUtils::read_config(map_name);
    std::pair<int, int> safe_dist{10, 10};

    EgoParams ego_params = TestUtils::build_ego_params(conf);
    SearchParams search_params = TestUtils::build_search_params(conf, /*gpu=*/true);

    TestUtils::export_color_frame(conf, "output.png");

    SearchFrame *frame = search_params.frame();
    frame->processDistanceToGoal(search_params.goal().x(), search_params.goal().z());
    frame->processSafeDistanceZone(safe_dist, /*computeVectorized=*/true);
    frame->setVehicleParams(5.25, angle::deg(40));
    frame->setPhysicalDimensionInMeters(200, 200);
    frame->processKinematicExclusionAreas(search_params.start(), search_params.goal());

    FastRRT planner(ego_params, /*smartExpansion=*/true);
    planner.setPlanData(search_params);

    while (true)
    {
        exec_local_planner(planner, *frame, conf, /*max_timeout_improving=*/-1);
    }
}

int main()
{
    exec_fastrrt_costmap("map_cost_25");
    return 0;
}
