#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <driveless/search_frame.h>
#include <driveless/cuda_basic.h>
#include <driveless/angle.h>
#include "test_utils.h"
#include "../../include/wpmp_graph.h"

TEST(TestWGraphExpansion, GraphExpansion)
{
    SearchFrame *frame = createEmptySearchFrame(800, 800, {-1, -1}, {-1, -1});

    for (int z = 300; z < 310; z++)
        for (int x = 300; x < frame->width(); x++)
        {
            frame->set({x, z}, {5, 0, 0});
        }

    WGraph graph(frame);

    graph.clear();
    // graph.set_start(50, 99, 0);

    // graph.set_start(128, 255, 0);

    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
        {1},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    frame->setClassCosts(costs);
    frame->setClassColors({{0, 0, 0},
                           {128, 0, 128},
                           {128, 128, 128},
                           {0, 128, 128},
                           {128, 128, 0},
                           {255, 255, 255}});

    frame->setPhysicalDimensionInMeters(80, 80);
    frame->setVehicleParams(5.412658773, maxSteering);

    Waypoint origin(400, 799, angle::rad(0));
    Waypoint goal(400, 0, angle::rad(0));
    graph.set_start(origin.x(), origin.z(), origin.heading().rad());
    graph.compute_goal_wave(frame, goal);
    auto mat = exportGraph(frame, &graph, "export_output.png");

    graph.expand(frame, 400, 799, 300, goal, 20, angle::deg(5).rad());

    graph.expand(frame, 400, 600, 300, goal, 20, angle::deg(5).rad());

    graph.expand(frame, 312, 437, 300, goal, 20, angle::deg(5).rad());

    graph.expand(frame, 214, 526, 300, goal, 20, angle::deg(5).rad());

    auto mat2 = exportGraph(frame, &graph, "export_output.png");

    //graph.connect_to_goal()
}