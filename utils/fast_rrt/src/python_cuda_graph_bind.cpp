#include "../include/graph.h"
#include "../../cudac/include/cuda_frame.h"
#include "../include/fastrrt.h"
extern "C"
{
    void *cudagraph_initialize(
        int width,
        int height,
        float perceptionWidthSize_m,
        float perceptionHeightSize_m,
        float maxSteeringAngle_deg,
        float vehicleLength,
        int minDistance_x, int minDistance_z,
        int lowerBound_x, int lowerBound_z,
        int upperBound_x, int upperBound_z)
    {
        CudaGraph *g = new CudaGraph(width, height);
        g->setSearchParams({minDistance_x, minDistance_z},
            {lowerBound_x, lowerBound_z},
            {upperBound_x, upperBound_z});
        g->setPhysicalParams(perceptionWidthSize_m, perceptionHeightSize_m, angle::deg(maxSteeringAngle_deg), vehicleLength);
        g->setClassCosts((int *)segmentationClassCost, 29);
        return g;
    }
    void cudagraph_destroy(void *ptr)
    {
        CudaGraph *graph = (CudaGraph *)ptr;
        delete graph;
    }


    void compute_apf(void *ptr, void *cudaFramePtr, int radius) {
        CudaGraph *graph = (CudaGraph *)ptr;
        CudaFrame *frame = (CudaFrame *)cudaFramePtr;

        graph->computeAPF(frame->getFramePtr(), radius);
    }

    float * get_intrinsic_costs (void *ptr) {
        CudaGraph *graph = (CudaGraph *)ptr;
        float3 * frameData = graph->getFrameDataPtr()->getCudaPtr();

        int width = graph->width();
        int height = graph->height();

        float * data = new float[width * height];

        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++) {
                int i = h * width + w;
                data[i] = frameData[i].z;
            }

        return data;
    }

    void destroy_intrinsic_costs_ptr (float *ptr) {
        delete []ptr;
    }
}