#pragma once

#ifndef __ANGLE_DRIVELESS_H
#define __ANGLE_DRIVELESS_H

#include "math_utils.h"
#include <iostream>

// CODE:BEGIN

class angle
{
    double _val;

private:
    angle(double val) : _val(val) {}

public:
    static angle deg(double val);
    static angle rad(double val);
    inline double deg() const { return TO_DEG * _val; }
    inline double rad() const { return _val; }
    void setDeg(double val);
    void setRad(double val);

    // Compound
    inline angle operator+(const angle &other) { return angle::rad(_val + other._val); }
    inline angle operator-(const angle &other) { return angle::rad(_val - other._val); }

    template <typename T>
    inline angle operator+(const T other) { return angle::rad(_val + other); }
    template <typename T>
    inline angle operator-(const T other) { return angle::rad(_val - other); }
    template <typename T>
    inline angle operator/(const T other) { return angle::rad(_val / other); }
    template <typename T>
    inline angle operator*(const T other) { return angle::rad(_val * other); }

    // Binary
    inline bool operator==(const angle &other) const
    {
        double v1 = _val;
        double v2 = other._val;
        if (v2 >= 0 && v2<HALF_PI & v1> Q3_INIT & v1 < DOUBLE_PI)
            return __TOLERANCE_EQUALITY(abs(v1 - DOUBLE_PI), v2);

        if (v1 >= 0 & v1<HALF_PI & v2> Q3_INIT & v2 < DOUBLE_PI)
            return __TOLERANCE_EQUALITY(abs(v2 - DOUBLE_PI), v1);

        return __TOLERANCE_EQUALITY(_val, other._val);
    }
    inline bool operator!=(const angle &other) { return !(__TOLERANCE_EQUALITY(_val, other._val)); }
    inline bool operator<(const angle &other) { return __TOLERANCE_LOWER(_val, other._val); }
    inline bool operator>(const angle &other) { return __TOLERANCE_GREATER(_val, other._val); }
    inline bool operator<=(const angle &other) { return __TOLERANCE_EQUALITY(_val, other._val) || __TOLERANCE_LOWER(_val, other._val); }
    inline bool operator>=(const angle &other) { return __TOLERANCE_EQUALITY(_val, other._val) || __TOLERANCE_GREATER(_val, other._val); }

    friend std::ostream &operator<<(std::ostream &os, const angle &a)
    {
        os << "(deg: " << a.deg() << ", rad: " << a.rad() << ")";
        return os;
    }
};

// CODE:END

#endif