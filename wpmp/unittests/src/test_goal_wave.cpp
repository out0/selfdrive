#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <driveless/search_frame.h>
#include <driveless/cuda_basic.h>
#include "test_utils.h"
#include "../../include/wpmp_graph.h"

TEST(TestWGraph, GoalWaveClear)
{
    SearchFrame * frame = createEmptySearchFrame(800, 800, {-1, -1}, {-1, -1});

    WGraph graph(800, 800);

    graph.clear();
    graph.set_start(400, 780, 0);

    angle maxSteering = angle::deg(40);
    std::vector<float> costs = {
        {1},
        {1},
        {2},
        {3},
        {4},
        {-1}};

    frame->setClassCosts(costs);
    frame->setClassColors({
        {0, 0, 0},
        {128, 0, 128},
        {128, 128, 128},
        {0, 128, 128},
        {128, 128, 0},
        {255, 255, 255}
    });

    graph.set_physical_params(800, 800, maxSteering, 15.412658773);
    graph.set_search_params({0, 0}, {-1, -1}, {-1, -1});
    graph.set_frame_class_costs(costs);

    Waypoint goal(400, 0, angle::rad(0));


    printf("processing safe distance check\n");
    frame->processSafeDistanceZone({10, 10}, false);

    printf("computing goal wave\n");
    graph.compute_goal_wave(frame, goal);

    exportGraph(frame, &graph, "output.png");

    delete frame;
}