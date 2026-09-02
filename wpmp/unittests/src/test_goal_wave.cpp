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
    // SearchFrame * frame = createEmptySearchFrame(100, 100, {-1, -1}, {-1, -1}, {8, 8});
    SearchFrame *frame = createEmptySearchFrame(800, 800, {-1, -1}, {-1, -1});

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

    Waypoint origin(400, 0, angle::rad(0));
    graph.set_start(origin.x(), origin.z(), origin.heading().rad());
    Waypoint goal(400, 0, angle::rad(0));
    // Waypoint goal(128, 0, angle::rad(0));

    printf("processing safe distance check\n");
    frame->processSafeDistanceZone({5, 5}, false);
    frame->processDistanceToGoal(goal.x(), goal.z());

    showSearchParameters(frame);

    timeIt("exclusion areas", [&]() { //
        frame->processKinematicExclusionAreas(origin, goal);
    });

    std::vector<uchar> dest(static_cast<size_t>(frame->width()) * frame->height() * 3);
    frame->exportToColorFrame(dest.data());
    cv::Mat cimg(frame->height(), frame->width(), CV_8UC3, dest.data());
    cv::imwrite("frame.png", cimg);

    timeIt("goal wave", [&]() { //
        graph.compute_goal_wave(frame, goal);
    });

    exportGraph(frame, &graph, "output.png");

    delete frame;
}