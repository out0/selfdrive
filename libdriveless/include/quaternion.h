#pragma once

#ifndef __QUATERNION_DRIVELESS_H
#define __QUATERNION_DRIVELESS_H

#include "angle.h"
#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include "cuda_ptr.h"

#if defined(CUDA_VERSION_MAJOR) && CUDA_VERSION_MAJOR >= 13
#define double4 double4_16a
#endif

__device__ __host__ double quaternion_size_sq(double4 *p);
__device__ __host__ void quaternion_multiply(double4 *store, double4 *p, double4 *q);
__device__ __host__ double quaternion_size(double4 *p);
__device__ __host__ void quaternion_invert(double4 *store, double4 *p);
__device__ __host__ void quaternion_conjugate(double4 *store, double4 *p);
__device__ __host__ void quaternion_divide(double4 *store, double4 *p, double4 *q);
__device__ __host__ void quaternion_rotate(double4 *store, double4 *p, double4 *q);
__device__ __host__ void quaternion_rotate_x(double4 *store, double4 *p, double angle_rad);
__device__ __host__ void quaternion_rotate_y(double4 *store, double4 *p, double angle_rad);
__device__ __host__ void quaternion_rotate_z(double4 *store, double4 *p, double angle_rad);
__device__ __host__ double quaternion_angle_to_axis(double4 *p, double4 *axis, bool is_neg, bool is_unitary);
__device__ __host__ bool quaternion_equals(const double4 *p, const double4 *q);
__device__ __host__ void quaternion_sum(double4 *store, const double4 *p, const double4 *q);
__device__ __host__ void quaternion_minus(double4 *store, const double4 *p, const double4 *q);



class quaternion
{
    double4 * data;
    bool __is_unitary;
    bool __is_owner;

private:
    static quaternion multiply(quaternion *self, const quaternion *other);
    static quaternion divide(quaternion *numerator, quaternion *denominator);

public:
    quaternion();
    quaternion(double w, double x, double y, double z);
    quaternion(angle yaw_x, angle pitch_y, angle roll_x);
    quaternion(double4 *data);
    ~quaternion();

    double w() const { return data->w; }
    double x() const { return data->x; }
    double y() const { return data->y; }
    double z() const { return data->z; }

    

    inline void set(double w, double x, double y, double z) {
        data->w = w;
        data->x = x;
        data->y = y;
        data->z = z;
    }

    inline quaternion operator+(const quaternion &other) const {
        quaternion q;
        quaternion_sum(q.data, data, other.data);
        return q;
    }

    template <typename T>
    inline quaternion operator+(T other) const { 
        return quaternion(data->w + other, data->x, data->y, data->z); 
    }

    inline quaternion operator-(const quaternion &other) const { 
        quaternion q;
        quaternion_minus(q.data, data, other.data);
        return q;
    }
    template <typename T>
    inline quaternion operator-(T other) const { 
        return quaternion(data->w - other, data->x, data->y, data->z); 
    }

    
    inline quaternion operator/(quaternion &other) { 
        return quaternion::divide(this, &other); 
    }

    inline quaternion operator*(quaternion &other)
    {
        return quaternion::multiply(this, &other);
    }
    inline quaternion operator*(const quaternion &other)
    {
        return quaternion::multiply(this, &other);
    }

    quaternion operator-() const {
        return quaternion(
            0.0 - data->w, 
            0.0 - data->x,
            0.0 - data->y,
            0.0 - data->z);
    }

    template <typename T>
    inline quaternion operator*(T other) const
    {
        return quaternion(data->w * other, data->x * other, data->y * other, data->z * other);
    }
    inline bool operator==(const quaternion &other) const
    {
//        printf ("oper equal1 (%.2f, %.2f, %.2f, %.2f) == (%.2f, %.2f, %.2f, %.2f)\n", data->w, data->x, data->y, data->z, other.w(), other.x(), other.y(), other.z());
        return quaternion_equals(data, other.data);
    }
    inline bool operator==(quaternion &other) const
    {
  //      printf ("oper equal2\n");
        return quaternion_equals(data, other.data);
    }

    friend std::ostream& operator<<(std::ostream& os, const quaternion& q) {
        os << q.to_string();
        return os;
    }

    quaternion invert() const;
    quaternion clone() const;
    double size() const;

    angle yaw() const;
    angle pitch() const;
    angle roll() const;

    void rotate_x(angle a);
    void rotate_y(angle a);
    void rotate_z(angle a);

    // Yaw (ψ) → Rotation around the Z-axis (left or right, like turning a car).
    inline void rotate_yaw(angle a)
    {
        rotate_z(a);
    }
    // Pitch (θ) → Rotation around the Y-axis (tilting up or down, like nodding).
    inline void rotate_pitch(angle a)
    {
        rotate_y(a);
    }
    // Roll (φ) → Rotation around the X-axis (tilting side to side, like a plane rolling).
    inline void rotate_roll(angle a)
    {
        rotate_x(a);
    }

    std::string to_string() const;

    double4 * get_data() {
        return data;
    }
};
#endif

