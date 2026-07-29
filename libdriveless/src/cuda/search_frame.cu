#include "../../include/search_frame.h"
#include "../../include/cuda_basic.h"
#include <stdexcept>

extern __global__ void __CUDA_SetGoal(float3 *frame, float *classCosts, int *_searchSpaceParams, int half_minDist_px, int goal_x, int goal_z);
extern __device__ __host__ bool __computeFeasibleForAngle(float3 *frame, int *params, float *classCost, int minDistX, int minDistZ, int x, int z, float angle_radians);

SearchFrame::SearchFrame(
    int width, int height, 
    std::pair<int, int> lowerBound, 
    std::pair<int, int> upperBound, 
    std::pair<int, int> searchZoneDim, 
    int numCPUThreadHandlers) : Frame<float3>(width, height, numCPUThreadHandlers), _numCPUThreadHandlers(numCPUThreadHandlers), _searchZoneDim(searchZoneDim)
{
    // _classColors = nullptr;
    // _classCosts = nullptr;
    // _params = nullptr;
    // _bestValue = nullptr;
    _classCount = 0;
    _safeZoneChecked = false;
    _safeZoneVectorialChecked = false;
    _distanceToGoalProcessed = false;

    _params = std::make_unique<CudaPtr<int>>(11);
    _bestValue = std::make_unique<CudaPtr<int>>(1);

    _params->get()[FRAME_PARAM_WIDTH] = width;
    _params->get()[FRAME_PARAM_HEIGHT] = height;
    _params->get()[FRAME_PARAM_LOWER_BOUND_X] = lowerBound.first;
    _params->get()[FRAME_PARAM_LOWER_BOUND_Z] = lowerBound.second;
    _params->get()[FRAME_PARAM_UPPER_BOUND_X] = upperBound.first;
    _params->get()[FRAME_PARAM_UPPER_BOUND_Z] = upperBound.second;
    _params->get()[FRAME_PARAM_CENTER_X] = TO_INT(width / 2);
    _params->get()[FRAME_PARAM_CENTER_Z] = TO_INT(height / 2);
    _params->get()[FRAME_PARAM_MIN_DIST_X] = 0;
    _params->get()[FRAME_PARAM_MIN_DIST_Z] = 0;
}

SearchFrame::~SearchFrame()
{
}



void SearchFrame::clear()
{
    Frame::clear();
    _safeZoneChecked = false;
    _safeZoneVectorialChecked = false;
}

void SearchFrame::copyFrom(float *ptr)
{
    Frame::copyFrom(ptr);
    _safeZoneChecked = false;
    _safeZoneVectorialChecked = false;
}

void SearchFrame::copyTo(float *ptr)
{
    long pos_cuda = 0;
    long pos_outp = 0;
    const int W = width();
    const int H = height();

    float3 *cuda_ptr = getPtr();
    for (int z = 0; z < H; z++)
        for (int x = 0; x < W; x++)
        {
            pos_cuda = z * W + x;
            pos_outp = 3 * pos_cuda;
            ptr[pos_outp] = cuda_ptr[pos_cuda].x;
            ptr[pos_outp + 1] = cuda_ptr[pos_cuda].y;
            ptr[pos_outp + 2] = cuda_ptr[pos_cuda].z;
        }
}

void SearchFrame::setClassCosts(std::vector<float> classCosts)
{
    _classCount = classCosts.size();
    _classCosts = std::make_unique<CudaPtr<float>>(_classCount);

    int i = 0;
    for (auto val : classCosts)
    {
        _classCosts->get()[i] = val;
        i++;
    }
}

std::vector<float> SearchFrame::getClassCosts() {
    std::vector<float> res;
    
    if (_classCount == 0)
        return res;

    res.reserve(_classCount);
    
    for (int i = 0; i < _classCount; i++) {
        res.push_back(_classCosts->get()[i]);
    }
    return res;
}

float SearchFrame::getClassCost(unsigned int classId)
{
    if (_classCosts == nullptr)
        throw std::invalid_argument("you need to first call setClassCosts()");

    if (classId >= _classCount)
    {
        throw std::invalid_argument("parameter " + std::to_string(classId) + "is bigger than the last class id: " + std::to_string(_classCount - 1));
    }

    return _classCosts->get()[classId];
}

float SearchFrame::getClassCostForNode(int x, int z)
{
    int classId = static_cast<int>(at({x, z}).x);
    return getClassCost(classId);
}
float SearchFrame::getClassCostForNode(long pos)
{
    int classId = static_cast<int>(getPtr()[pos].x);
    return getClassCost(classId);
}

bool SearchFrame::isObstacle(int x, int z)
{
    int classId = static_cast<int>(at({x, z}).x);
    return _classCosts->get()[classId] < 0;
}

bool SearchFrame::isTraversable(int x, int z)
{
    int traversability = static_cast<int>(at({x, z}).z);
    return (traversability & 0x100) > 0;
}

inline bool check_bit (int traversability, int bit) {
    return traversability & bit > 0;
}

int SearchFrame::getTraversability(int x, int z) {
    return static_cast<int>(at({x, z}).z);
}

bool SearchFrame::isTraversable(int x, int z, angle heading, bool precision_check)
{

    if (_safeZoneChecked || _safeZoneVectorialChecked) {
        int traversability = static_cast<int>(at({x, z}).z);

        // all angle traversable bit
        if (check_bit(traversability, 0x100)) return true;

        if (_safeZoneVectorialChecked) {
            auto pair = checkTraversableAngleBitPairCheck(heading.rad());

            if (pair.second == -1)
                return check_bit(traversability, pair.first);
            
            if (!precision_check)
                return check_bit(traversability, pair.first) && check_bit(traversability, pair.second);
        }
    }

    int min_dist_x = _params->get()[FRAME_PARAM_MIN_DIST_X];
    int min_dist_z = _params->get()[FRAME_PARAM_MIN_DIST_Z];

    return __computeFeasibleForAngle(getPtr(), getFrameParamsPtr(), getCudaClassCostsPtr(), min_dist_x, min_dist_z, x, z, heading.rad());
}

double SearchFrame::getCost(int x, int z)
{
    return at({x, z}).y;
}

double SearchFrame::computeVehicleLength(double perceptionHeightSize_m)
{
    double dz = abs(_params->get()[FRAME_PARAM_LOWER_BOUND_Z] - _params->get()[FRAME_PARAM_UPPER_BOUND_Z]);

    return 0.5 * perceptionHeightSize_m * dz / height();
}

void SearchFrame::setValues(int x, int z, float v1, float v2, float v3) {
    
}