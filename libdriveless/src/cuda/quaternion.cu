#include "../../include/quaternion.h"
#include <cmath>
#include <array>

__device__ __host__ double quaternion_size_sq(double4 *p)
{
    return p->w * p->w + p->x * p->x + p->y * p->y + p->z * p->z;
}
__device__ __host__ void quaternion_multiply(double4 *store, double4 *p, double4 *q)
{
    store->w = p->w * q->w - p->x * q->x - p->y * q->y - p->z * q->z + 0.0;
    store->x = p->w * q->x + p->x * q->w + p->y * q->z - p->z * q->y + 0.0;
    store->y = p->w * q->y - p->x * q->z + p->y * q->w + p->z * q->x + 0.0;
    store->z = p->w * q->z + p->x * q->y - p->y * q->x + p->z * q->w + 0.0;
}
__device__ __host__ double quaternion_size(double4 *p)
{
    return sqrtf(p->w * p->w + p->x * p->x + p->y * p->y + p->z * p->z);
}
__device__ __host__ void quaternion_invert(double4 *store, double4 *p)
{
    double s = 1 / quaternion_size_sq(p);
    store->w = (p->w * s) + 0.0;
    store->x = -1 * (p->x * s) + 0.0;
    store->y = -1 * (p->y * s) + 0.0;
    store->z = -1 * (p->z * s) + 0.0;
}
__device__ __host__ void quaternion_conjugate(double4 *store, double4 *p)
{
    store->w = p->w + 0.0;
    store->x = -p->x + 0.0;
    store->y = -p->y + 0.0;
    store->z = -p->z + 0.0;
}
__device__ __host__ void quaternion_divide(double4 *store, double4 *p, double4 *q)
{
    double4 tmp;
    quaternion_invert(&tmp, q);
    quaternion_multiply(store, p, &tmp);
}
__device__ __host__ void quaternion_rotate(double4 *store, double4 *p, double4 *q)
{
    double4 tmp, q_c;
    quaternion_multiply(&tmp, q, p);
    quaternion_conjugate(&q_c, q);
    quaternion_multiply(store, &tmp, &q_c);
}
__device__ __host__ void quaternion_rotate_x(double4 *store, double4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    double4 v{s, 0, 0, c}; // double4 convention puts w at the end
    quaternion_rotate(store, p, &v);
}
__device__ __host__ void quaternion_rotate_y(double4 *store, double4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    double4 v{0, s, 0, c};
    quaternion_rotate(store, p, &v);
}
__device__ __host__ void quaternion_rotate_z(double4 *store, double4 *p, double angle_rad)
{
    double a = angle_rad * 0.5;
    double c = cos(a);
    double s = sin(a);
    double4 v{0, 0, s, c};
    quaternion_rotate(store, p, &v);
}
__device__ __host__ double quaternion_angle_to_axis(double4 *p, double4 *axis, bool is_neg, bool is_unitary)
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
__device__ __host__ bool quaternion_equals(const double4 *p, const double4 *q)
{
    return __TOLERANCE_EQUALITY(p->w, q->w) &&
           __TOLERANCE_EQUALITY(p->x, q->x) &&
           __TOLERANCE_EQUALITY(p->y, q->y) &&
           __TOLERANCE_EQUALITY(p->z, q->z);
}
__device__ __host__ void quaternion_sum(double4 *store, const double4 *p, const double4 *q)
{
    store->w = p->w + q->w;
    store->x = p->x + q->x;
    store->y = p->y + q->y;
    store->z = p->z + q->z;
}
__device__ __host__ void quaternion_minus(double4 *store, const double4 *p, const double4 *q)
{
    store->w = p->w - q->w;
    store->x = p->x - q->x;
    store->y = p->y - q->y;
    store->z = p->z - q->z;
}
/// @brief Unit quaternion
/// @param data
quaternion::quaternion()
{
    this->data = new double4{0, 0, 0, 1};
    __is_unitary = true;
    __is_owner = true;
}

/// @brief Quaternion init
/// @param data
quaternion::quaternion(double w, double x, double y, double z)
{
    this->data = new double4{x, y, z, w};
    __is_unitary = quaternion_size_sq(data) == 1;
    __is_owner = true;
}

/// @brief Quaternion manipulation of existing data. The data is not going to be freed.
/// @param data
quaternion::quaternion(double4 *val)
{
    this->data = val;
    __is_unitary = quaternion_size_sq(val) == 1;
    __is_owner = false;
}
/// @brief Quaternion initialization by Euler angles, rotating the X-aix following the convention ZYX
/// @param yaw_z
/// @param pitch_y
/// @param roll_x
quaternion::quaternion(angle yaw_z, angle pitch_y, angle roll_x)
{
    this->data = new double4{1, 0, 0, 0};
    this->data->x = 1.0;
    __is_unitary = true;
    __is_owner = true;
    rotate_z(yaw_z);
    rotate_y(pitch_y);
    rotate_x(roll_x);
}

quaternion::~quaternion()
{
    if (this->data && this->__is_owner)
    {
        delete this->data;
    }
}

double quaternion::size() const
{
    return quaternion_size(data);
}

quaternion quaternion::invert()  const
{
    quaternion q;
    double4 *p = q.data;
    quaternion_invert(p, data);
    return q;
}

quaternion quaternion::clone()  const
{
    return quaternion(data->w, data->x, data->y, data->z);
}

quaternion quaternion::divide(quaternion *numerator, quaternion *denominator)
{
    quaternion res;
    quaternion_divide(res.data, numerator->data, denominator->data);
    return res;
}

quaternion quaternion::multiply(quaternion *self, const quaternion *other)
{
    quaternion q;
    quaternion_multiply(q.data, self->data, other->data);
    return q;
}

/*
      z
      |
      |
      |
      |
      +------------ x
     /
    /
   /
   y

    Yaw = the angle that the new vector v makes with the aix X or Y. We chose X, thus  A = <v, x>. To support a > 180, we check if A < 0
    v = (x, y, z) has neg A when   v x Z = (y, -x, 0) with y < 0.

    Pitch = the angle that the new vector v makes with the aix X or Z. We chose X, thus  A = <v, x>. To support a > 180, we check if A < 0
    v = (x, y, z) has neg A when   v x Y = (-z, 0, x) with z > 0.

    Roll = the angle that the new vector v makes with the aix Y or Z. We chose Y, thus  A = <v, Y>. To support a > 180, we check if A < 0
    v = (x, y, z) has neg A when   v x X = (0, z, -y) with z < 0.

*/

angle quaternion::yaw() const
{
    double4 axis = {1, 0, 0, 0}; // obs: X axis, double4 convention puts w in the end
    bool neg = data->y < 0;
    return angle::rad(quaternion_angle_to_axis(data, &axis, neg, __is_unitary));
}

angle quaternion::pitch() const
{
    double4 axis = {1, 0, 0, 0}; // obs: X axis, double4 convention puts w in the end
    bool neg = data->z > 0;
    return angle::rad(quaternion_angle_to_axis(data, &axis, neg, __is_unitary));
}

angle quaternion::roll() const
{
    double4 axis = {0, 1, 0, 0}; // obs: Y axis, double4 convention puts w in the end
    bool neg = data->z < 0;
    return angle::rad(quaternion_angle_to_axis(data, &axis, neg, __is_unitary));
}

void quaternion::rotate_x(angle a)
{
    quaternion_rotate_x(data, data, a.rad());
}

void quaternion::rotate_y(angle a)
{
    quaternion_rotate_y(data, data, a.rad());
}

void quaternion::rotate_z(angle a)
{
    quaternion_rotate_z(data, data, a.rad());
}

std::string quaternion::to_string() const
{
    char buffer[50];
    snprintf(buffer, sizeof(buffer), "(%.6f, %.6f, %.6f, %.6f)", data->w, data->x, data->y, data->z);
    return std::string(buffer);
}

extern "C"
{
    void *p__quaternion_new(double w, double x, double y, double z)
    {
        return new quaternion(w, x, y, z);
    }
    void p__quaternion_destroy(void *p)
    {
        quaternion *q = (quaternion *)p;
        delete q;
    }
    void *p__quaternion_sum_scalar(void *p, double q)
    {
        quaternion *self = (quaternion *)p;
        return new quaternion(self->w() + q, self->x(), self->y(), self->z());
    }
    void *p__quaternion_sum(void *p, void *q)
    {
        quaternion *outp = new quaternion();
        quaternion_sum(outp->get_data(), ((quaternion *)p)->get_data(), ((quaternion *)q)->get_data());
        return outp;
    }
    void *p__quaternion_minus_scalar(void *p, double q)
    {
        quaternion *self = (quaternion *)p;
        return new quaternion(self->w() - q, self->x(), self->y(), self->z());
    }
    void *p__quaternion_minus(void *p, void *q)
    {
        quaternion *outp = new quaternion();
        quaternion_minus(outp->get_data(), ((quaternion *)p)->get_data(), ((quaternion *)q)->get_data());
        return outp;
    }
    void *p__quaternion_mul_scalar(void *p, double q)
    {
        quaternion *self = (quaternion *)p;
        return new quaternion(self->w() * q, self->x() * q, self->y() * q, self->z() * q);
    }
    void *p__quaternion_mul(void *p, void *q)
    {
        quaternion *outp = new quaternion();
        quaternion_multiply(outp->get_data(), ((quaternion *)p)->get_data(), ((quaternion *)q)->get_data());
        return outp;
    }
    void *p__quaternion_div_scalar(void *p, double q)
    {
        quaternion *self = (quaternion *)p;
        return new quaternion(self->w() / q, self->x() / q, self->y() / q, self->z() / q);
    }
    void *p__quaternion_div(void *p, void *q)
    {
        quaternion *outp = new quaternion();
        quaternion_divide(outp->get_data(), ((quaternion *)p)->get_data(), ((quaternion *)q)->get_data());
        return outp;
    }
    bool p__quaternion_equals(void *p, void *q)
    {
        return quaternion_equals(((quaternion *)p)->get_data(), ((quaternion *)q)->get_data());
    }

    void *p__quaternion_invert(void *p)
    {
        quaternion *outp = new quaternion();
        quaternion_invert(outp->get_data(), ((quaternion *)p)->get_data());
        return outp;
    }
    void *p__quaternion_conjugate(void *p)
    {
        quaternion *outp = new quaternion();
        quaternion_conjugate(outp->get_data(), ((quaternion *)p)->get_data());
        return outp;
    }
    double p__quaternion_size(void *p)
    {
        return ((quaternion *)p)->size();
    }

    void p__rotate_yaw(void *p, double angle_rad)
    {
        ((quaternion *)p)->rotate_yaw(angle::rad(angle_rad));
    }
    void p__rotate_pitch(void *p, double angle_rad)
    {
        ((quaternion *)p)->rotate_pitch(angle::rad(angle_rad));
    }
    void p__rotate_roll(void *p, double angle_rad)
    {
        ((quaternion *)p)->rotate_roll(angle::rad(angle_rad));
    }
    double p__yaw(void *p)
    {
        return ((quaternion *)p)->yaw().rad();
    }
    double p__pitch(void *p)
    {
        return ((quaternion *)p)->pitch().rad();
    }
    double p__roll(void *p)
    {
        return ((quaternion *)p)->roll().rad();
    }
    double * p__value(void *p)
    {
        auto q = (quaternion *)p;
        double *res = new double[4];
        res[0] = q->w();
        res[1] = q->x();
        res[2] = q->y();
        res[3] = q->z();
        return res;
    }
    void p__value_free(double *ptr)
    {
        delete []ptr;
    }
    
}