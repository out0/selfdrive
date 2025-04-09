
#include "../../../cudac/include/cuda_basic.h"
#include "../../include/cuda_params.h"
#include "../../include/graph.h"
#include <cstdlib>
#include <ctime>

__device__ __host__ long computePos(int width, int x, int z)
{
    return z * width + x;
}

__device__ __host__ bool set(int4 *graph, float3 *graphData, long pos, float heading, int parent_x, int parent_z, float cost, int type, bool override)
{
#ifdef __CUDA_ARCH__
    if (override)
    {
        atomicExch(&(graph[pos].z), type);
    }
    else if (!atomicCAS(&(graph[pos].z), 0, type) == GRAPH_TYPE_NULL)
        return false;
#else
    if (!override && graph[pos].z != GRAPH_TYPE_NULL)
    {
        return false;
    }
    graph[pos].z = type;
#endif

    // will return if z is originally not 0.
    graph[pos].x = parent_x;
    graph[pos].y = parent_z;
    graphData[pos].x = heading;
    graphData[pos].y = cost;
    return true;
}


__device__ __host__ void setParentCuda(int4 *graph, long pos, int parent_x, int parent_z)
{
    graph[pos].x = parent_x;
    graph[pos].y = parent_z;
}

__device__ __host__ int2 getParentCuda(int4 *graph, long pos)
{
    return {graph[pos].x, graph[pos].y};
}

__device__ __host__ void setTypeCuda(int4 *graph, long pos, int type)
{
    graph[pos].z = type;
}

__device__ __host__ int getTypeCuda(int4 *graph, long pos)
{
    return graph[pos].z;
}

__device__ __host__ void incNodeDeriveCount(int4 *graph, long pos)
{
    graph[pos].w++;
}

__device__ __host__ int getNodeDeriveCount(int4 *graph, long pos)
{
    return graph[pos].w;
}

__device__ __host__ float getHeadingCuda(float3 *graphData, long pos)
{
    return graphData[pos].x;
}

__device__ __host__ inline void setHeadingCuda(float3 *graphData, long pos, float heading)
{
    graphData[pos].x = heading;
}

__device__ __host__ float getCostCuda(float3 *graphData, long pos)
{
    return graphData[pos].y;
}

__device__ __host__ inline void setCostCuda(float3 *graphData, long pos, float cost)
{
    graphData[pos].y = cost;
}


__device__ __host__ float getIntrinsicCostCuda(float3 *graphData, long pos)
{
    return graphData[pos].z;
}

__device__ __host__ float getIntrinsicCost(float3 *graphData, int width, int x, int z)
{
    long pos = computePos(width, x, z);
    return graphData[pos].z;
}

__device__ __host__ void setIntrinsicCostCuda(float3 *graphData, long pos, float cost)
{
    graphData[pos].z = cost;
}
__device__ __host__ void setIntrinsicCost(float3 *graphData, int width, int x, int z, float cost)
{
    long pos = computePos(width, x, z);
    graphData[pos].z = cost;
}
__device__  void incIntrinsicCost(float3 *graphData, int width, int x, int z, float cost)
{
    long pos = computePos(width, x, z);
    atomicAdd(&graphData[pos].z, cost);
}

__device__ __host__ bool checkInGraphCuda(int4 *graph, long pos)
{
    return graph[pos].z == GRAPH_TYPE_NODE;
}



void CudaGraph::setType(int x, int z, int type)
{
    long pos = computePos(_frame->width(), x, z);
    setTypeCuda(_frame->getCudaPtr(), pos, type);
}


CudaGraph::CudaGraph(int width, int height)
{
    _frame = std::make_shared<CudaGrid<int4>>(width, height);
    _frameData = std::make_unique<CudaGrid<float3>>(width, height);
    if (!cudaAllocMapped(&this->_parallelCount, sizeof(unsigned int)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(unsigned int)) + std::string(" bytes for counting\n");
        throw msg;
    }

    _gridCenter.x = TO_INT(width / 2);
    _gridCenter.y = TO_INT(height / 2);
    _physicalParams = nullptr;
    //_searchSpaceParams = nullptr;

    if (!cudaAllocMapped(&this->_searchSpaceParams, 10*sizeof(int)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(11*sizeof(int)) + std::string(" bytes for search-space parameters\n");
        throw msg;
    }

    _searchSpaceParams[FRAME_PARAM_WIDTH] = width;
    _searchSpaceParams[FRAME_PARAM_HEIGHT] = height;
    _searchSpaceParams[FRAME_PARAM_CENTER_X] = _gridCenter.x;
    _searchSpaceParams[FRAME_PARAM_CENTER_Z] = _gridCenter.y;


    // TODO: make this method refresh randomness for each clear() in graph
    __initializeRandomGenerator();

    if (!cudaAllocMapped(&this->_goalReached, sizeof(bool)))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(bool)) + std::string(" bytes for goal reached check\n");
        throw msg;
    }


}
CudaGraph::~CudaGraph()
{
    cudaFreeHost(_parallelCount);
}

void CudaGraph::setPhysicalParams(float perceptionWidthSize_m, float perceptionHeightSize_m, angle maxSteeringAngle, float vehicleLength)
{

    if (!cudaAllocMapped(&this->_physicalParams, sizeof(double) * 8))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(double) * 5) + std::string(" bytes for physical params\n");
        throw msg;
    }

    this->_physicalParams[PHYSICAL_PARAMS_RATE_W] = _frame->width() / perceptionWidthSize_m;
    this->_physicalParams[PHYSICAL_PARAMS_INV_RATE_W] = perceptionWidthSize_m / _frame->width();
    this->_physicalParams[PHYSICAL_PARAMS_RATE_H] = _frame->height() / perceptionHeightSize_m;
    this->_physicalParams[PHYSICAL_PARAMS_INV_RATE_H] = perceptionHeightSize_m / _frame->height();
    this->_physicalParams[PHYSICAL_PARAMS_MAX_STEERING_RAD] = maxSteeringAngle.rad();
    this->_physicalParams[PHYSICAL_PARAMS_MAX_STEERING_DEG] = maxSteeringAngle.deg();
    this->_physicalParams[PHYSICAL_PARAMS_LR] = vehicleLength / 2;
}

void CudaGraph::setSearchParams(std::pair<int, int> minDistance, std::pair<int, int> lowerBound, std::pair<int, int> upperBound) {
    _searchSpaceParams[FRAME_PARAM_MIN_DIST_X] = TO_INT((float)minDistance.first / 2);
    _searchSpaceParams[FRAME_PARAM_MIN_DIST_Z] = TO_INT((float)minDistance.second / 2);
    _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_X] = lowerBound.first;
    _searchSpaceParams[FRAME_PARAM_LOWER_BOUND_Z] = lowerBound.second;
    _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_X] = upperBound.first;
    _searchSpaceParams[FRAME_PARAM_UPPER_BOUND_Z] = upperBound.second;
}

void CudaGraph::setClassCosts(const int *costs, int size) {
    
    if (!cudaAllocMapped(&this->_classCosts, sizeof(float) * size))
    {
        std::string msg = "[CUDA GRAPH] unable to allocate memory with " + std::to_string(sizeof(float) * size) + std::string(" bytes for class cost list\n");
        throw msg;
    }

    for (int i = 0; i < size; i++) {
        this->_classCosts[i] = static_cast<float>(costs[i]);
    }
}

void CudaGraph::addStart(int x, int z, angle heading)
{
    add(x, z, heading, -1, -1, 0);
}

void CudaGraph::add(int x, int z, angle heading, int parent_x, int parent_z, float cost)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_frame->width(), x, z);
    set(_frame->getCudaPtr(), _frameData->getCudaPtr(), pos, heading.rad(), parent_x, parent_z, cost, GRAPH_TYPE_NODE, true);
}
void CudaGraph::addTemporary(int x, int z, angle heading, int parent_x, int parent_z, float cost)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_frame->width(), x, z);
    set(_frame->getCudaPtr(), _frameData->getCudaPtr(), pos, heading.rad(), parent_x, parent_z, cost, GRAPH_TYPE_TEMP, true);
}

bool CudaGraph::__checkLimits(int x, int z)
{
    if (x < 0 || x >= _frame->width())
        return false;
    if (z < 0 || z >= _frame->height())
        return false;

    return true;
}

void CudaGraph::remove(int x, int z)
{
    if (!__checkLimits(x, z))
        return;
    setTypeCuda(_frame->getCudaPtr(), computePos(_frame->width(), x, z), GRAPH_TYPE_NULL);
}

__global__ static void __CUDA_KERNEL_clear(int4 *graph, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    graph[pos].z = GRAPH_TYPE_NULL;
}

void CudaGraph::clear()
{
    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    __CUDA_KERNEL_clear<<<numBlocks, THREADS_IN_BLOCK>>>(_frame->getCudaPtr(), width(), height());

    CUDA(cudaDeviceSynchronize());
    *_goalReached = false;
}

bool CudaGraph::checkInGraph(int x, int z)
{
    if (!__checkLimits(x, z))
        return false;

    long pos = computePos(_frame->width(), x, z);
    return checkInGraphCuda(_frame->getCudaPtr(), pos);
}

void CudaGraph::setParent(int x, int z, int parent_x, int parent_z)
{
    if (!__checkLimits(x, z))
        return;
    long pos = computePos(_frame->width(), x, z);
    setParentCuda(_frame->getCudaPtr(), pos, parent_x, parent_z);
}

int2 CudaGraph::getParent(int x, int z)
{
    if (!__checkLimits(x, z) || getType(x, z) == GRAPH_TYPE_NULL)
        return {-1, -1};

    long pos = computePos(_frame->width(), x, z);
    return getParentCuda(_frame->getCudaPtr(), pos);
}

angle CudaGraph::getHeading(int x, int z)
{
    long pos = computePos(_frameData->width(), x, z);
    return angle::rad(getHeadingCuda(_frameData->getCudaPtr(), pos));
}

void CudaGraph::setHeading(int x, int z, angle heading)
{
    if (!__checkLimits(x, z))
        return;

    long pos = computePos(_frameData->width(), x, z);
    setHeadingCuda(_frameData->getCudaPtr(), pos, heading.rad());
}

float CudaGraph::getCost(int x, int z)
{
    if (!__checkLimits(x, z))
        return -1;

    long pos = computePos(_frameData->width(), x, z);
    return getCostCuda(_frameData->getCudaPtr(), pos);
}
void CudaGraph::setCost(int x, int z, float cost)
{
    if (!__checkLimits(x, z))
        return;

    long pos = computePos(_frameData->width(), x, z);
    setCostCuda(_frameData->getCudaPtr(), pos, cost);
}

int CudaGraph::getType(int x, int z)
{
    if (!__checkLimits(x, z))
        return -1;

    long pos = computePos(_frame->width(), x, z);
    return getTypeCuda(_frame->getCudaPtr(), pos);
}

void CudaGraph::dumpGraph(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    int4 * fptr = _frame->getCudaPtr();
    float3 * fptrData = _frameData->getCudaPtr();

    for (int z = 0; z < _frame->height(); z++) {
        for (int x = 0; x < _frame->width(); x++) {
            long pos = z * _frame->width() + x;
            fprintf(fp, "%d %d %d %f %f %f\n", fptr[pos].x, fptr[pos].y, fptr[pos].z, 
                    fptrData[pos].x, fptrData[pos].y, fptrData[pos].z);
        }
    }

    fclose(fp);
    printf("Graph dumped to %s\n", filename);
}

void CudaGraph::readfromDump(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    int4 * fptr = _frame->getCudaPtr();
    float3 * fptrData = _frameData->getCudaPtr();

    for (int z = 0; z < _frame->height(); z++) {
        for (int x = 0; x < _frame->width(); x++) {
            long pos = z * _frame->width() + x;
            fscanf(fp, "%d %d %d %f %f %f\n", &fptr[pos].x, &fptr[pos].y, &fptr[pos].z, 
                    &fptrData[pos].x, &fptrData[pos].y, &fptrData[pos].z);
        }
    }

    fclose(fp);
    printf("Graph read from %s\n", filename);
}