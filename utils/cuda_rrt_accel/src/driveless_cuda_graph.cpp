
#include <string>
#include <memory>
#include <cstring>

#include "../include/cuda_basic.h"
#include "../include/cuda_graph.h"
#include "../include/class_def.h"

extern void CUDA_clear(float4 *frame, int width, int height);
extern void CUDA_list_neighbors(int radius, int *list, int *count);
extern int *CUDA_find_best_neighbor(float4 *frame, int3 *point, int width, int height, int goal_x, int goal_z, float radius);
extern int *CUDA_find_nearest_neighbor(float4 *frame, int3 *point, int width, int height, int goal_x, int goal_z);
extern int CUDA_count_elements_in_graph(float4 *frame, int width, int height);
extern bool CUDA_check_in_graph(float4 *frame, int width, int height, int x, int z);
extern void CUDA_link(float4 *frame, 
    float3 *cuda_frame,
    int width,
    int height,
    int *classCosts,
    int min_dist_x, 
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z,    
    int parent_x,
    int parent_z,
    int x,
    int z);

extern void CUDA_list_elements(float4 *frame, 
    float *result,
    int width,
    int height,
    int count);

CudaGraph::CudaGraph(int width, int height,
                     int min_dist_x,
                     int min_dist_z,
                     int lower_bound_ego_x,
                     int lower_bound_ego_z,
                     int upper_bound_ego_x,
                     int upper_bound_ego_z)
{
    this->width = width;
    this->height = height;
    this->min_dist_x = min_dist_x;
    this->min_dist_z = min_dist_z;
    this->lower_bound_ego_x = lower_bound_ego_x;
    this->lower_bound_ego_z = lower_bound_ego_z;
    this->upper_bound_ego_x = upper_bound_ego_x;
    this->upper_bound_ego_z = upper_bound_ego_z;

    if (!cudaAllocMapped(&this->frame, sizeof(float4) * (width * height)))
        return;

    if (!cudaAllocMapped(&this->point, sizeof(int3)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for point output\n", sizeof(int2) * 2);
        cudaFreeHost(frame);
        return;
    }

    int *costs = nullptr;
    const int numClasses = 29;
    if (!cudaAllocMapped(&costs, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for class costs\n", sizeof(int) * numClasses);
        cudaFreeHost(frame);
        cudaFreeHost(point);
        return;
    }
    for (int i = 0; i < numClasses; i++)
        costs[i] = segmentationClassCost[i];

    clear();
}

CudaGraph::~CudaGraph()
{
    cudaFreeHost(this->frame);
    cudaFreeHost(this->point);
}

void CudaGraph::clear()
{
    CUDA_clear(this->frame, this->width, this->height);
}

int *CudaGraph::find_nearest_neighbor(int x, int z)
{
    return CUDA_find_nearest_neighbor(frame, point, width, height, x, z);
}

int *CudaGraph::find_best_neighbor(int x, int z, float radius)
{
    return CUDA_find_best_neighbor(frame, point, width, height, x, z, radius);
}

void CudaGraph::add_point(int x, int z, int parent_x, int parent_z, float cost)
{
    int pos = width * z + x;
    this->frame[pos].x = parent_x;
    this->frame[pos].y = parent_z;
    this->frame[pos].z = cost;
    this->frame[pos].w = 1.0;
}

unsigned int CudaGraph::count()
{
    return CUDA_count_elements_in_graph(frame, width, height);
    // unsigned int value = atomicInc(&count, gridDim.x);
}
bool CudaGraph::checkInGraph(int x, int z)
{
    return CUDA_check_in_graph(frame, width, height, x, z);
}

int *CudaGraph::getParent(int x, int z)
{
    int pos = width * z + x;

    if (this->frame[pos].w == 0.0)
        return new int[3]{0, 0, 0};

    return new int[3]{
        (int)this->frame[pos].x,
        (int)this->frame[pos].y,
        1};
}

void CudaGraph::link(float3 *cuda_frame, int parent_x, int parent_z, int x, int z)
{
    CUDA_link(frame, 
            cuda_frame,
            width, 
            height, 
            classCosts,
            min_dist_x,
            min_dist_z,
            lower_bound_ego_x,
            lower_bound_ego_z,
            upper_bound_ego_x,
            upper_bound_ego_z,            
            parent_x, parent_z, x, z);

}


void CudaGraph::listNodes(float *res, int count) {
    CUDA_list_elements(frame, res, width, height, count);
}

// int CudaGraph::listGraphPoints(void *self, int *points)
// {
//     int len = count();
//     points = new int[len * 3];
//     int p = 0;

//     for (int z = 0; z < height; z++) {
//         for (int x = 0; x < width; x++) {
//             int pos = width * z + x;

//             if (this->frame[pos].w != 1.0)
//                 continue;

//             points[p] =
//         }
//     }
// }