#include "../include/cpu_frame.h"

void copy_data(float *ptr, float4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<float>(ptr[posPtr + 3]);
}
void copy_data(float *ptr, double4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<double>(ptr[posPtr + 3]);
}
void copy_data(float *ptr, int4 *dest, long pos)
{
    long posPtr = 4 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
    dest[pos].w = static_cast<int>(ptr[posPtr + 3]);
}
void copy_data(float *ptr, float3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<float>(ptr[posPtr + 2]);
}
void copy_data(float *ptr, double3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<double>(ptr[posPtr]);
    dest[pos].y = static_cast<double>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<double>(ptr[posPtr + 2]);
}
void copy_data(float *ptr, int3 *dest, long pos)
{
    long posPtr = 3 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
    dest[pos].z = static_cast<int>(ptr[posPtr + 2]);
}
void copy_data(float *ptr, float2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<float>(ptr[posPtr]);
    dest[pos].y = static_cast<float>(ptr[posPtr + 1]);
}
void copy_data(float *ptr, double2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = ptr[posPtr];
    dest[pos].y = ptr[posPtr + 1];
}
void copy_data(float *ptr, int2 *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos].x = static_cast<int>(ptr[posPtr]);
    dest[pos].y = static_cast<int>(ptr[posPtr + 1]);
}
void copy_data(float *ptr, float *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos] = static_cast<float>(ptr[posPtr]);
}
void copy_data(float *ptr, double *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos] = static_cast<double>(ptr[posPtr]);
}
void copy_data(float *ptr, int *dest, long pos)
{
    long posPtr = 2 * pos;
    dest[pos] = static_cast<int>(ptr[posPtr]);
}

