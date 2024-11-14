#include "cuda_graph.h"
#include "class_def.h"

extern void CUDA_clear(double4 *graph, double *graph_cost, int width, int height);
extern unsigned int CUDA_parallel_count(double4 *graph, unsigned int *pcount, int width, int height);
extern bool CUDA_check_connection_feasible(float3 *og, int *classCost, double *checkParams, unsigned int *pcount, double3 &start, double3 &end);
extern void __tst_CUDA_build_path(double4 *graph, float3 *og, double *checkParams, double3 &start, double3 &end, int r, int g, int b);
extern int2 CUDA_find_nearest_neighbor(double4 *graph, int3 *point, int *bestValue, int width, int height, int x, int z);
extern int2 CUDA_find_nearest_feasible_neighbor(double4 *graph, float3 *og, int *classCost, double *checkParams, int3 *point, int *bestValue, int x, int z);
extern void CUDA_list_elements(double4 *graph, double *graph_cost, double *result, int width, int height, int count);

CudaGraph::CudaGraph(
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z,
    double _rate_w,
    double _rate_h,
    double _max_steering_angle_deg,
    double _lr,
    double velocity_meters_per_s)
{
    this->width = width;
    this->height = height;

    if (!cudaAllocMapped(&this->graph, sizeof(double4) * (width * height)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate graph with %ld bytes\n", sizeof(double4) * (width * height));
        return;
    }

    if (!cudaAllocMapped(&this->graph_cost, sizeof(double) * (width * height)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate graph_cost with %ld bytes\n", sizeof(double) * (width * height));
        cudaFreeHost(graph);
        return;
    }

    if (!cudaAllocMapped(&this->point, sizeof(int3)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for fast point output computing\n", sizeof(int2) * 2);
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        return;
    }

    classCosts = nullptr;
    const int numClasses = 29;
    if (!cudaAllocMapped(&classCosts, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for class costs\n", sizeof(int) * numClasses);
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        return;
    }
    for (int i = 0; i < numClasses; i++)
        classCosts[i] = segmentationClassCost[i];

    if (!cudaAllocMapped(&pcount, sizeof(unsigned int)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for counting tasks\n", sizeof(unsigned int));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        return;
    }

    if (!cudaAllocMapped(&bestValue, sizeof(int)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for computing best value tasks\n", sizeof(int));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        cudaFreeHost(pcount);
        return;
    }

    if (!cudaAllocMapped(&checkParams, 15 * sizeof(double)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for the computing params\n", 30 * sizeof(double));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        cudaFreeHost(pcount);
        cudaFreeHost(bestValue);
        return;
    }

    checkParams[0] = width;
    checkParams[1] = height;
    checkParams[2] = min_dist_x;
    checkParams[3] = min_dist_z;
    checkParams[4] = lower_bound_ego_x;
    checkParams[5] = lower_bound_ego_z;
    checkParams[6] = upper_bound_ego_x;
    checkParams[7] = upper_bound_ego_z;
    checkParams[8] = _rate_w;
    checkParams[9] = _rate_h;
    checkParams[10] = _max_steering_angle_deg;
    checkParams[11] = _lr;
    checkParams[12] = velocity_meters_per_s;
    checkParams[13] = width / 2;  // OG center
    checkParams[14] = height / 2; // OG center

    clear();
}

CudaGraph::~CudaGraph()
{
    cudaFreeHost(this->graph);
    cudaFreeHost(this->graph_cost);
    cudaFreeHost(this->point);
    cudaFreeHost(this->classCosts);
    cudaFreeHost(this->pcount);
    cudaFreeHost(this->bestValue);
    cudaFreeHost(this->checkParams);
}

void CudaGraph::clear()
{
    CUDA_clear(this->graph, this->graph_cost, this->width, this->height);
}

void CudaGraph::add(int x, int z, double heading, int parent_x, int parent_z, double cost)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;

    this->graph[pos].x = parent_x;
    this->graph[pos].y = parent_z;
    this->graph[pos].z = heading;
    this->graph[pos].w = 1.0;
    this->graph_cost[pos] = cost;
}

void CudaGraph::remove(int x, int z)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;

    this->graph[pos].w = 0.0;
}

unsigned int CudaGraph::count()
{
    return CUDA_parallel_count(graph, pcount, width, height);
}

bool CudaGraph::checkInGraph(int x, int z)
{
    if (x > width || x < 0)
        return false;
    if (z > height || z < 0)
        return false;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return false;

    return this->graph[pos].w == 1.0;
}

void CudaGraph::setParent(int x, int z, int parent_x, int parent_z)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;
    if (this->graph[pos].w == 0.0)
        return;

    this->graph[pos].x = parent_x;
    this->graph[pos].y = parent_z;
}
int2 CudaGraph::getParent(int x, int z)
{
    int2 p;
    p.x = -1;
    p.y = -1;

    if (x > width || x < 0)
        return p;
    if (z > height || z < 0)
        return p;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return p;
    if (this->graph[pos].w == 0.0)
        return p;

    p.x = this->graph[pos].x;
    p.y = this->graph[pos].y;
    return p;
}

void CudaGraph::setCost(int x, int z, double cost)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;
    int pos = z * width + x;
    if (pos > width * height || pos < 0)
        return;
    if (this->graph[pos].w == 0.0)
        return;

    this->graph_cost[pos] = cost;
}
double CudaGraph::getCost(int x, int z)
{
    if (x > width || x < 0)
        return -1;
    if (z > height || z < 0)
        return -1;
    int pos = z * width + x;
    if (pos > width * height || pos < 0)
        return -1;
    if (this->graph[pos].w == 0.0)
        return -1;

    return this->graph_cost[pos];
}

void CudaGraph::list(double *result, int count)
{
    CUDA_list_elements(graph, graph_cost, result, width, height, count);
}

bool CudaGraph::checkConnectionFeasible(float3 *og, double3 &start, double3 end)
{
    return CUDA_check_connection_feasible(og, classCosts, checkParams, pcount, start, end);
}

void CudaGraph::drawKinematicPath(float3 *og, double3 &start, double3 &end)
{
    __tst_CUDA_build_path(graph, og, checkParams, start, end, 255, 255, 255);
}

int2 CudaGraph::find_nearest_neighbor(int x, int z)
{
    return CUDA_find_nearest_neighbor(graph, point, bestValue, width, height, x, z);
}

int2 CudaGraph::find_nearest_feasible_neighbor(float3 *og, int x, int z)
{
    return CUDA_find_nearest_feasible_neighbor(graph, og, classCosts, checkParams, point, bestValue, x, z);
}