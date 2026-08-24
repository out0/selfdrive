#include <gtest/gtest.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <driveless/search_frame.h>
#include <driveless/cuda_basic.h>
#include <driveless/angle.h>
#include "test_utils.h"
#include "../../include/wpmp_graph.h"

TEST(TestWGraph, GoalWaveClear)
{
    SearchFrame * frame = createEmptySearchFrame(100, 100, {-1, -1}, {-1, -1}, {8, 8});
    //SearchFrame * frame = createEmptySearchFrame(256, 256, {-1, -1}, {-1, -1});

    WGraph graph(frame);

    graph.clear();
    graph.set_start(50, 99, 0);
    //graph.set_start(128, 255, 0);

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

    frame->setPhysicalDimensionInMeters(10, 10);
    frame->setVehicleParams(5.412658773, maxSteering);

    Waypoint goal(50, 0, angle::rad(0));
    //Waypoint goal(128, 0, angle::rad(0));


    printf("processing safe distance check\n");
    frame->processSafeDistanceZone({5, 5}, false);
    frame->processDistanceToGoal(goal.x(), goal.z());

    showSearchParameters(frame);

    printf("computing goal wave\n");

    auto start = std::chrono::high_resolution_clock::now();
    graph.compute_goal_wave(frame, goal);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Execution time: " << duration / 1000 << " ms" << " (" << duration << ") us" << std::endl;  

    exportGraph(frame, &graph, "output.png");

    delete frame;
}