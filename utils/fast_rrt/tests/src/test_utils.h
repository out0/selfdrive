#ifndef __TEST_UTILS_DRIVELESS_H
#define __TEST_UTILS_DRIVELESS_H

#include "../include/graph.h"

extern bool _ASSERT_DEQ(double a, double b, int tolerance = 4);
#define ASSERT_DEQ(a, b) ASSERT_TRUE(_ASSERT_DEQ(a, b))

void exportGraph(CudaGraph *graph, const char *filename);

float3 *createEmptySearchFrame(int width, int height);

void destroySearchFrame(float3 * ptr);

#endif