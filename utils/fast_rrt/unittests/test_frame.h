#ifndef H_TEST_FRAME
#define H_TEST_FRAME

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../src/cuda_frame.h"
#include "../src/cuda_graph.h"
#include "test_utils.h"
#include <unordered_set>

#define OG_REAL_WIDTH 34.641016151377535
#define OG_REAL_HEIGHT 34.641016151377535
#define MAX_STEERING_ANGLE 40

class TestFrame {
    CudaFrame *og;
    CudaGraph *graph;

private:
    int to_set_key(int x, int z) {
        return 1000 * x + z;
    }

    void drawNode(CudaGraph *graph, float3 *imgPtr, int x, int z, double heading) {
        int2 parent = graph->getParent(x, z);

        if (parent.x < 0 or parent.y < 0) return;

        double3 start, end;
        start.x = parent.x;
        start.y = parent.y;
        start.z = heading;
        end.x = x;
        end.y = z;
        end.z = heading;

        graph->drawKinematicPath(imgPtr, start, end);
    }

public:
    TestFrame(int default_fill_value = 3) {
        og = create_default_cuda_frame(default_fill_value);

        double rw = OG_WIDTH / OG_REAL_WIDTH ;
        double rh = OG_HEIGHT / OG_REAL_HEIGHT ;
        double lr = 0.5 * (LOWER_BOUND_Z - UPPER_BOUND_Z) / rh;

        graph = new CudaGraph(
            OG_WIDTH, 
            OG_HEIGHT, 
            MIN_DIST_X,
            MIN_DIST_Z, 
            LOWER_BOUND_X, 
            LOWER_BOUND_Z, 
            UPPER_BOUND_X, 
            UPPER_BOUND_Z, 
            rw, rh, 
            MAX_STEERING_ANGLE, 
            lr, 1.0);
    }

    ~TestFrame() {
        delete og;
        delete graph;
    }

    void toFile(const char * filename = "dump.png") {
        dump_cuda_frame_to_file(og, filename);
    }
    CudaGraph * getGraph() {
        return graph;
    }

    void drawGraph() {
        float3 * imgPtr = og->getFramePtr();

        int num_nodes = graph->count();
        if (num_nodes == 0) return;

        double *nodes = new double[6 * sizeof(double) * num_nodes];
        graph->list(nodes, num_nodes);

        std::unordered_set<int> drawn;

        for (int i = 0; i < num_nodes; i++) {
            int pos = 6 * i;
            int x = static_cast<int>(nodes[pos]);
            int z = static_cast<int>(nodes[pos+1]);
            int key = to_set_key(x, z);

            if (drawn.find(key) != drawn.end()) continue;

            double heading = static_cast<int>(nodes[pos+2]);
            drawNode(graph, imgPtr, x, z, heading);
            drawn.insert(key);
        }

        graph->drawNodes(imgPtr);
        
    }
};


#endif