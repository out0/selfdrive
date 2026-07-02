#include "../../include/quaternion.h"
#include <cmath>
#include <array>

__device__ __host__ double quaternion_size_sq(DOUBLE4 *p)
{
    return p->w * p->w + p->x * p->x + p->y * p->y + p->z * p->z;
}
__device__ __host__ void quaternion_multiply(DOUBLE4 *store, DOUBLE4 *p, DOUBLE4 *q)
{
    store->w = p->w * q->w - p->x * q->x - p->y * q->y - p->z * q->z + 0.0;
    store->x = p->w * q->x + p->x * q->w + p->y * q->z - p->z * q->y + 0.0;
    store->y = p->w * q->y - p->x * q->z + p->y * q->w + p->z * q->x + 0.0;
    store->z = p->w * q->z + p->x * q->y - p->y * q->x + p->z * q->w + 0.0;
}
__device__ __host__ double quaternion_size(DOUBLE4 *p)
{
    return sqrtf(p->w * p->w + p->x * p->x + p->y * p->y + p->z * p->z);
}
__device__ __host__ void quaternion_invert(DOUBLE4 *store, DOUBLE4 *p)
{
    double s = 1 / quaternion_size_sq(p);
    store->w = (p->w * s) + 0.0;
    store->x = -1 * (p->x * s) + 0.0;
    store->y = -1 * (p->y * s) + 0.0;
    store->z = -1 * (p->z * s) + 0.0;
}
__device__ __host__ void quaternion_conjugate(DOUBLE4 *store, DOUBLE4 *p)
{
    store->w = p->w + 0.0;
    store->x = -p->x + 0.0;
    store->y = -p->y + 0.0;
    store->z = -p->z + 0.0;
}
__device__ __host__ void quaternion_divide(DOUBLE4 *store, DOUBLE4 *p, DOUBLE4 *q)
{
    DOUBLE4 tmp;
    quaternion_invert(&tmp, q);
    quaternion_multiply(store, p, &tmp);
}
__device__ __host__ void quaternion_rotate(DOUBLE4 *store, DOUBLE4 *p, DOUBLE4 *q)
{
    DOUBLE4 tmp, q_c;
    quaternion_multiply(&tmp, q, p);
    quaternion_conjugate(&q_c, q);
    quaternion_multiply(store, &tmp, &q_c);
}
__device__ __host__ void quaternion_rotate_x(DOUBLE4 *store, DOUBLE4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    DOUBLE4 v{s, 0, 0, c}; // DOUBLE4 convention puts w at the end
    quaternion_rotate(store, p, &v);
}
__device__ __host__ void quaternion_rotate_y(DOUBLE4 *store, DOUBLE4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    DOUBLE4 v{0, s, 0, c};
    quaternion_rotate(store, p, &v);
}
__device__ __host__ void quaternion_rotate_z(DOUBLE4 *store, DOUBLE4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    DOUBLE4 v{0, 0, s, c};
    quaternion_rotate(store, p, &v);
}
__device__ __host__ double quaternion_angle_to_axis(DOUBLE4 *p, DOUBLE4 *axis, bool is_neg, bool is_unitary)
{
    double c = p->w * axis->w + p->x * axis->x + p->y * axis->y + p->z * axis->z;

    if (!is_unitary)
    {
        double sp = quaternion_size(p);
        c = c / sp;
    }
    if (is_neg)
    {
        return acos(-1 * c) + PI;
    }
    return acos(c);
}
__device__ __host__ bool quaternion_equals(const DOUBLE4 *p, const DOUBLE4 *q)
{
    return __TOLERANCE_EQUALITY(p->w, q->w) &&
           __TOLERANCE_EQUALITY(p->x, q->x) &&
           __TOLERANCE_EQUALITY(p->y, q->y) &&
           __TOLERANCE_EQUALITY(p->z, q->z);
}
__device__ __host__ void quaternion_sum(DOUBLE4 *store, const DOUBLE4 *p, const DOUBLE4 *q)
{
    store->w = p->w + q->w;
    store->x = p->x + q->x;
    store->y = p->y + q->y;
    store->z = p->z + q->z;
}
__device__ __host__ void quaternion_minus(DOUBLE4 *store, const DOUBLE4 *p, const DOUBLE4 *q)
{
    store->w = p->w - q->w;
    store->x = p->x - q->x;
    store->y = p->y - q->y;
    store->z = p->z - q->z;
}
