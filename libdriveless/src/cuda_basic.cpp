#include "../include/cuda_basic.h"

std::unique_ptr<float4[]> copyToCpuMemory(std::vector<Waypoint> points)
{
    int count = points.size();
    std::unique_ptr<float4[]> ptr = std::make_unique<float4[]>(count);
    float4 *addr = ptr.get();

    for (int c = 0; c < count; c++)
    {
        addr[c].x = points[c].x();
        addr[c].y = points[c].z();
        addr[c].z = points[c].heading().rad();
        addr[c].w = 0.0;
    }
    return ptr;
}

std::unique_ptr<float4[]> copyToCpuMemory(float *path, int count)
{
    std::unique_ptr<float4[]> ptr = std::make_unique<float4[]>(count);
    float4 *addr = ptr.get();

    for (int c = 0; c < count; c++)
    {
        int pos = 4 * c;
        addr[c].x = path[pos];
        addr[c].y = path[pos + 1];
        addr[c].z = path[pos + 2];
        addr[c].w = 0.0;
    }
    return ptr;
}
