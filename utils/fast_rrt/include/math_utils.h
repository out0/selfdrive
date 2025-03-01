#pragma once

#ifndef __MATH_UTILS_DRIVELESS_H
#define __MATH_UTILS_DRIVELESS_H

// CODE:BEGIN


#ifdef __CUDA_ARCH__
    // On GPU, use CUDA-specific rounding
    #define TO_INT(x) __double2int_rn(x)
#else
    // On CPU, use standard rounding
    #include <cmath>
    #define TO_INT(x) static_cast<int>(roundf(x))
#endif

//#define __EQUALITY_TOLERANCE 0.0001
#define __EQUALITY_TOLERANCE 0.001
#define __CLOSE_VALUE 0.01
#define __TOLERANCE_EQUALITY(a, b) std::fabs(a - b) <= __EQUALITY_TOLERANCE
#define __TOLERANCE_CLOSE_VALUE(a, b) std::fabs(a - b) <= __CLOSE_VALUE
#define __TOLERANCE_LOWER(a, b) a - b < __EQUALITY_TOLERANCE
#define __TOLERANCE_GREATER(a, b) a - b > __EQUALITY_TOLERANCE

#define DOUBLE_PI 6.2831853071795862e+0
#define PI 3.1415926535897931e+0
#define HALF_PI 1.5707963267948966e+0
#define QUARTER_PI 0.7853981633974483e+0
#define TO_RAD 0.017453293
#define TO_DEG 57.295779513
#define Q3_INIT 4.71238896803846898

// CODE:END



#endif