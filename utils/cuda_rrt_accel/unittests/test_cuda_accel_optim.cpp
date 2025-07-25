#include "../include/cuda_graph.h"
#include "../include/cuda_frame.h"
#include <gtest/gtest.h>

float compute_dist(int x1, int z1, int x2, int z2) {
    int dx = x2 - x1;
    int dz = z2 - z1;
    return sqrt(dx * dx + dz * dz);
}


float * buildWithValue(int width, int height, float value) {
    float *imgPtr = new float[width * height * 3];
    for (int i = 0; i < 1000; i++)
        for (int j = 0; j < 1000; j++) {
            int pos = 3 * (i * 1000 + j);
            imgPtr[pos] = value;
            imgPtr[pos+1] = value;
            imgPtr[pos+2] = value;
        }
    return imgPtr;
}


TEST(CudaRRTAccel, TestOptimization)
{
    CudaGraph g(1000, 1000, 0, 0, 0, 0, 0, 0);
    g.add_point(500, 500, -1, -1, 0);
    float cost = 0 + compute_dist(500, 450, 500, 500);
    g.add_point(500, 450, 500, 500, cost);
    cost += compute_dist(400, 350, 500, 450);
    g.add_point(400, 350, 500, 450, cost);
    cost += compute_dist(300, 250, 400, 350);
    g.add_point(300, 250, 400, 350, cost);
    cost += compute_dist(300, 250, 500, 0);
    g.add_point(500, 0, 300, 250, cost);

    ASSERT_EQ(g.count(), 5);

    int *parent = g.getParent(500, 0);
    ASSERT_EQ(parent[0], 300);
    ASSERT_EQ(parent[1], 250);
    delete []parent;

    // we add a new point, much better than 300, 250 as parent:
    g.add_point(500, 100, 500, 450, compute_dist(500, 100, 500, 450));

    // now we call for optimizing the graph with this point

    float *imgPtr = buildWithValue(1000, 1000, 1);
            
    CudaFrame * cudaFrame = new CudaFrame(imgPtr, 1000, 1000, 0, 0, 0, 0, 0, 0);
    cudaFrame->setGoalVectorized(1000, 1);

    g.optimizeGraph(cudaFrame->getFramePtr(), 500, 100, 500, 450, compute_dist(500, 100, 500, 450), 120);

    parent = g.getParent(500, 0);
    ASSERT_EQ(parent[0], 500);
    ASSERT_EQ(parent[1], 100);
    delete []parent;

}
