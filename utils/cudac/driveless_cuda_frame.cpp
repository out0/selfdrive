
#include <string>
#include <memory>
#include <cstring>

#include "cuda_basic.h"
#include "class_def.h"
#include "cuda_frame.h"

#define PATH_FEASIBLE_CPU_THRESHOLD 2000

extern uchar3 *CUDA_convertFrameColors(float3 *frame, int width, int height);

extern void CUDA_setGoal(
    float3 *frame,
    int width,
    int height,
    int goal_x,
    int goal_z,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z);

extern float4 *CUDA_checkFeasibleWaypoints(
    float *points,
    int count,
    float3 *frame,
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z);

extern void CUDA_setGoalVectorized(
    float3 *frame,
    int width,
    int height,
    int goal_x,
    int goal_z,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_x,
    int lower_bound_z,
    int upper_bound_x,
    int upper_bound_z);

void CudaFrame::copyToCpuPointer(float3 *source, float *target)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            int p = i * width + j;
            int f = 3 * p;
            target[f] = source[p].x;     // copy back the other two channels
            target[f + 1] = source[p].y; // copy back the other two channels
            target[f + 2] = source[p].z;
        }
}
void CudaFrame::copyToCpuPointer(uchar3 *source, u_char *target)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            int p = i * width + j;
            int f = 3 * p;
            target[f] = source[p].x;     // copy back the other two channels
            target[f + 1] = source[p].y; // copy back the other two channels
            target[f + 2] = source[p].z;
        }
}

CudaFrame::CudaFrame(float *ptr, int width, int height, int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z)
{
    this->width = width;
    this->height = height;
    this->min_dist_x = min_dist_x;
    this->min_dist_z = min_dist_z;
    this->lower_bound_x = lower_bound_x;
    this->lower_bound_z = lower_bound_z;
    this->upper_bound_x = upper_bound_x;
    this->upper_bound_z = upper_bound_z;

    if (!cudaAllocMapped(&this->frame, sizeof(float3) * (width * height)))
        return;

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int imgP = y * width + x;
            this->frame[imgP].x = ptr[3 * imgP];
        }
}

CudaFrame::~CudaFrame()
{
    cudaFreeHost(this->frame);
}

void CudaFrame::convertToColorFrame(u_char *dest)
{
    uchar3 *coloredFrame = CUDA_convertFrameColors(this->frame, this->width, this->height);
    copyToCpuPointer(coloredFrame, dest);
    cudaFreeHost(coloredFrame);
}

void CudaFrame::setGoal(int goal_x, int goal_z)
{
    CUDA_setGoal(this->frame,
                 this->width,
                 this->height,
                 goal_x,
                 goal_z,
                 this->min_dist_x,
                 this->min_dist_z,
                 this->lower_bound_x,
                 this->lower_bound_z,
                 this->upper_bound_x,
                 this->upper_bound_z);
}

void CudaFrame::copyBack(float *img)
{
    copyToCpuPointer(this->frame, img);
}

/// @brief Check which waypoints are feasible
/// @param points a four-position float array, where the first 3 channels represent x,z,heading pose and the last channel is to be rewritten carrying 1 for feasible or 0 for not-feasible
/// @param count how many points are in the array (total float * should be count * sizeof(float) * 4)
void CudaFrame::checkFeasibleWaypoints(float *points, int count)
{
    if (count == 0)
        return;

    if (count <= PATH_FEASIBLE_CPU_THRESHOLD)
    {
        checkFeasibleWaypointsCPU(
            points,
            count);
        return;
    }

    float4 *res = CUDA_checkFeasibleWaypoints(
        points,
        count,
        this->frame,
        this->width,
        this->height,
        this->min_dist_x,
        this->min_dist_z,
        this->lower_bound_x,
        this->lower_bound_z,
        this->upper_bound_x,
        this->upper_bound_z);

    if (res == nullptr)
        return;

    for (int c = 0; c < count; c++)
    {
        points[4 * c + 2] = res[c].z;
        points[4 * c + 3] = res[c].w;
    }

    cudaFreeHost(res);
}

void CudaFrame::setGoalVectorized(int goal_x, int goal_z)
{
    CUDA_setGoalVectorized(this->frame,
                           this->width,
                           this->height,
                           goal_x,
                           goal_z,
                           this->min_dist_x,
                           this->min_dist_z,
                           this->lower_bound_x,
                           this->lower_bound_z,
                           this->upper_bound_x,
                           this->upper_bound_z);
}
int CudaFrame::get_class_cost(int segmentation_class)
{
    return segmentationClassCost[segmentation_class];
}

static int __CPU_computeFeasibleForAngle(
    float3 *frame,
    int *classCost,
    int x,
    int z,
    float angle_radians,
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z)
{

    float c = cosf(angle_radians);
    float s = sinf(angle_radians);

    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = round(j * c - i * s) + x;
            int zl = round(j * s + i * c) + z;

            if (xl < 0 || xl >= width)
                continue;
            

            if (zl < 0 || zl >= height)
                continue;
            

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z)
                continue;
            

            int segmentation_class = round(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                return 0;
            }
        }

    return 1;
}

static float __CPU_compute_heading(float *waypoints, int pos1, int pos2, int waypoints_count, bool *valid, int width, int height)
{
    *valid = false;

    if (pos1 < 0 || pos2 < 0)
        return 0.0;

    if (pos1 > waypoints_count || pos2 > waypoints_count)
        return 0.0;

    int x1 = waypoints[pos1 * 4];
    int z1 = waypoints[pos1 * 4 + 1];
    int x2 = waypoints[pos2 * 4];
    int z2 = waypoints[pos2 * 4 + 1];

    if (x1 == x2 && z1 == z2)
        return 0.0;
    if (x1 < 0 || z1 < 0 || x2 < 0 || z2 < 0)
        return 0.0;
    if (x1 >= width || z1 >= height || x2 >= width || z2 >= height)
        return 0.0;

    float dx = x2 - x1;
    float dz = z2 - z1;
    *valid = true;
    float heading = 3.141592654F / 2 - atan2f(-dz, dx);

    if (heading > 3.141592654F) // greater than 180 deg
        heading = heading - 2 * 3.141592654F;

    return heading;
}

#define NUM_POINTS_ON_MEAN 3

static float __CPU__compute_mean_heading(float *waypoints, int pos, int waypoints_count, bool *valid, int width, int height)
{
    float heading = 0.0;
    int count = 0;

    for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
    {
        bool v = false;
        if (pos + j >= waypoints_count) 
            break;
        heading += __CPU_compute_heading(waypoints, pos, pos + j, waypoints_count, &v, width, height);
        if (!v)
            break;
        count++;
    }

    if (count != NUM_POINTS_ON_MEAN)
    {
        count = 0;
        // compute in reverse
        for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
        {
            bool v = false;
            if (pos - j < 0)  {
                *valid = false;
                return 0.0;
            }
            heading += __CPU_compute_heading(waypoints, pos - j, pos, waypoints_count, &v, width, height);
            if (!v)
                break;
            count++;
        }
    }

    *valid = count > 0;

    if (*valid)
        return heading / count;

    return 0.0;
}

void CudaFrame::checkFeasibleWaypointsCPU(float *waypoints, int count)
{
    for (int i = 0; i < count; i++)
    {
        int pos = 4 * i;
        int x = waypoints[pos];
        int z = waypoints[pos + 1];

        waypoints[pos + 3] = 1;

        if (x >= this->lower_bound_x && x <= this->upper_bound_x && z >= this->upper_bound_z && z <= this->lower_bound_z)
        {
            // printf("Inbound (%d, %d)\n", x, z);
            continue;
        }

        bool valid = false;
        float heading = __CPU__compute_mean_heading(waypoints, i, count, &valid, width, height);

        waypoints[pos + 2] = heading;

        if (!valid)
            continue;

        // if (x == 131 && z == 45)
        //     printf("pos = %d, x = %d, z = %d computed heading = %f\n", i, x, z, heading);

        // printf("CPU computing mean heading for (%d, %d) = %f\n", x, z, 180 * heading / 3.141592654F);

        waypoints[pos + 3] = __CPU_computeFeasibleForAngle(this->frame, (int *)segmentationClassCost, x, z, heading,
                                                           this->width, this->height, this->min_dist_x / 2, this->min_dist_z / 2, this->lower_bound_x, this->lower_bound_z,
                                                           this->upper_bound_x, this->upper_bound_z);
    }
}