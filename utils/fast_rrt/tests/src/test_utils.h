#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include "../include/graph.h"

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))

std::vector<int2> get_planned_path(CudaGraph *graph, float3 *ptr, angle goal_heading, int goal_x, int goal_z, float distToGoalTolerance);
float3 *createEmptySearchFrame(int width, int height);

void exportGraph(CudaGraph *graph, const char *filename, std::vector<int2> *path = nullptr);

float3 *createEmptySearchFrame(int width, int height);

void destroySearchFrame(float3 * ptr);

#endif