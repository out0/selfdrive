
#include <string>
#include <memory>
#include <cstring>

#include "cuda_basic.h"
#include "class_def.h"
#include "cuda_frame.h"

#define PATH_FEASIBLE_CPU_THRESHOLD 20

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

    // computeHeadings(points, count);

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

    if (x == 122 && z == 32) {
        printf("(%d, %d) heading: %f\n", x, z, (180 * angle_radians)/3.141592654);
    }


    for (int i = -min_dist_z; i <= min_dist_z; i++)
        for (int j = -min_dist_x; j <= min_dist_x; j++)
        {
            int xl = round(j * c - i * s) + x;
            int zl = round(j * s + i * c) + z;

        if (x == 122 && z == 32)
            printf("checking: (%d, %d) -> ", xl, zl);

            if (xl < 0 || xl >= width) {
                if (x == 122 && z == 32)
                    printf("oob\n");
                continue;
            }

            if (zl < 0 || zl >= height){
                if (x == 122 && z == 32)
                    printf("oob\n");
                continue;
            }

            if (xl >= lower_bound_ego_x && xl <= upper_bound_ego_x && zl >= upper_bound_ego_z && zl <= lower_bound_ego_z){
                if (x == 122 && z == 32)
                    printf("inbound\n");
                continue;
            }

            int segmentation_class = round(frame[zl * width + xl].x);

            if (classCost[segmentation_class] < 0)
            {
                // if (x == 131 && z == 45)
                //     printf("(%d, %d) not feasible because of position: (%d, %d) min_dist_x = %d \n", x, z, xl, z, min_dist_x);

                // if (x == DEBUG_X && z == DEBUG_Z)
                // {
                //     printf("(%d, %d) not feasible because of position: (%d, %d)\n", x, z, xl, zl);
                // }
                if (x == 122 && z == 32)
                    printf("not feasible\n");
                return 0;
            }

            if (x == 122 && z == 32)
                printf("feasible\n");
        }

    return 1;
}

/*
void CudaFrame::checkFeasibleWaypointsCPU(float *waypoints, int count)
{
    for (int i = 0; i < count; i++)
    {
        int pos = 4 * i;
        float heading = 0; // waypoints[pos + 2];

        float dz;
        float dx;
        float x1 = waypoints[4 * i];
        float z1 = waypoints[4 * i + 1];
        float x2 = waypoints[4 * (i + 1)];
        float z2 = waypoints[4 * (i + 1) + 1];

        if (i == 0)
        {
            dz = z2 - z1;
            dx = x2 - x1;
        }
        else
        {
            float x0 = waypoints[4 * (i - 1)];
            float z0 = waypoints[4 * (i - 1) + 1];
            if (i == count - 1)
            {
                dz = z1 - z0;
                dx = x1 - x0;
            } else {
                dz = z2 - z0;
                dx = x2 - x0;
            }
        }

        heading = 3.141592654F / 2 - atan2f(-dz, dx);
        waypoints[pos + 2] = heading;
        waypoints[pos + 3] = 1;

        if (x1 == 134 && z1 == 14) {
            printf("heading = %f\n", heading);
        }

        if (x1 >= this->lower_bound_x && x1 <= this->upper_bound_x && z1 >= this->upper_bound_z && z1 <= this->lower_bound_z)
            continue;

        waypoints[pos + 3] = __CPU_computeFeasibleForAngle(this->frame, (int *)segmentationClassCost, x1, z1, heading,
                                                           this->width, this->height, this->min_dist_x, this->min_dist_z, this->lower_bound_x, this->lower_bound_z,
                                                           this->upper_bound_x, this->upper_bound_z);
    }
}*/

static float __CPU_compute_heading(float *waypoints, int pos1, int pos2, bool *valid, int width, int height)
{
    *valid = false;

    if (pos1 < 0 || pos2 < 0)
        return 0.0;
    
    int x1 = waypoints[pos1 * 4];
    int z1 = waypoints[pos1 * 4 + 1];
    int x2 = waypoints[pos2 * 4];
    int z2 = waypoints[pos2 * 4 + 1];

    if (x1 == x2 && z1 == z2) return 0.0;
    if (x1 < 0 || z1 < 0 || x2 < 0 || z2 < 0) return 0.0;
    if (x1 >= width || z1 >= height || x2 >= width || z2 >= height) return 0.0;

    float dx = x2 - x1;
    float dz = z2 - z1;
    *valid = true;
    float heading = 3.141592654F / 2 - atan2f(-dz, dx);

    if (heading > 3.141592654F) // greater than 180 deg
        heading = heading - 2*3.141592654F;

    return heading;
}

#define NUM_POINTS_ON_MEAN 3

static float __CPU__compute_mean_heading(float *waypoints, int pos, bool *valid, int width, int height)
{
    float heading = 0.0;
    int count = 0;

    for (int j = 0; j < NUM_POINTS_ON_MEAN; j++)
    {
        bool v = false;
        heading += __CPU_compute_heading(waypoints, pos - j - 1, pos, &v, width, height);
        if (v)
            count++;
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

        printf(">>> %d = (%d, %d)\n", i, x, z);

        if (x == 122 && z == 32) {
            int k = 1;
            printf("%d (%d, %d) -> (%d, %d) (-%d)\n", i, x, z, (int)waypoints[4*(i - k)], (int)waypoints[4*(i - k) + 1], k);
            k = 2;
            printf("%d (%d, %d) -> (%d, %d) (-%d)\n", i, x, z, (int) waypoints[4*(i - k)], (int)waypoints[4*(i - k) + 1], k);
            k = 3;
            printf("%d (%d, %d) -> (%d, %d) (-%d)\n", i, x, z, (int)waypoints[4*(i - k)], (int)waypoints[4*(i - k) + 1], k);
        }

        waypoints[pos + 3] = 1;

        if (x >= this->lower_bound_x && x <= this->upper_bound_x && z >= this->upper_bound_z && z <= this->lower_bound_z) {
            // printf("Inbound (%d, %d)\n", x, z);
            continue;
        }

        bool valid = false;

        float heading = __CPU__compute_mean_heading(waypoints, i, &valid, width, height);
       

        if (z == 32)
        printf("CPU computed heading for (%d, %d) = %f\n", x, z, heading);

        if (!valid) {
            if (z == 32)
            printf("INVALID\n");
            continue;
        }

        // if (x == 131 && z == 45)
        //     printf("pos = %d, x = %d, z = %d computed heading = %f\n", i, x, z, heading);

        // printf("CPU computing mean heading for (%d, %d) = %f\n", x, z, 180 * heading / 3.141592654F);

        waypoints[pos + 3] = __CPU_computeFeasibleForAngle(this->frame, (int *)segmentationClassCost, x, z, heading,
                                                           this->width, this->height, this->min_dist_x / 2, this->min_dist_z / 2, this->lower_bound_x, this->lower_bound_z,
                                                           this->upper_bound_x, this->upper_bound_z);
    }
}

// void CudaFrame::computeHeadings(float *points, int count)
// {
//     points[3] = 0.0;

//     for (int i = 1; i < count - 1; i++)
//     {
//         int pos = 4 * (i - 1);
//         int pos2 = 4 * (i + 1);

//         float x1 = points[pos];
//         float z1 = points[pos + 1];
//         float x2 = points[pos2];
//         float z2 = points[pos2 + 1];
//         points[pos2 + 3] = __CPU_compute_heading(x1, z1, x2, z2);
//     }

//     points[4 * (count - 1) + 2] = points[4 * (count - 2) + 2];
// }