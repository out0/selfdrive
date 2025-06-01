
#pragma once

#ifndef H_CUDA_BASIC
#define H_CUDA_BASIC


/*#define HEADING_0 0x01
#define HEADING_22_5 0x02
#define HEADING_45 0x04
#define HEADING_67_5 0x08
#define HEADING_90 0x10
#define HEADING_MINUS_22_5 0x20
#define HEADING_MINUS_45 0x40
#define HEADING_MINUS_67_5 0x80*/

#define HEADING_0 0x80
#define HEADING_22_5 0x40
#define HEADING_45 0x20
#define HEADING_67_5 0x10
#define HEADING_90 0x08
#define HEADING_MINUS_22_5 0x04
#define HEADING_MINUS_45 0x02
#define HEADING_MINUS_67_5 0x01



#define ANGLE_HEADING_0 0.0
#define ANGLE_HEADING_22_5 CUDART_PI_F / 8
#define ANGLE_HEADING_45 CUDART_PI_F / 4
#define ANGLE_HEADING_67_5 (3*CUDART_PI_F) / 8
#define ANGLE_HEADING_90 CUDART_PI_F / 2
#define ANGLE_HEADING_MINUS_22_5 -CUDART_PI_F / 8
#define ANGLE_HEADING_MINUS_45 -CUDART_PI_F / 4
#define ANGLE_HEADING_MINUS_67_5 -(3*CUDART_PI_F) / 8

#define TOP 8       // 1000
#define BOTTOM 4    // 0100
#define LEFT 2      // 0010
#define RIGHT  1    // 0001
#define INSIDE 0    // 0000 

#define THREADS_IN_BLOCK 256

#include <driveless/cuda_utils.h>

#endif