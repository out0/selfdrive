#pragma once

#ifndef __MOVING_OBSTACLE_DRIVELESS_H
#define __MOVING_OBSTACLE_DRIVELESS_H

#include "state.h"
#include "cuda_ptr.h"
#include <vector>
#include <memory>

class MovingObstacle : public State
{
    int _width_px;
    int _height_px;

    int copyToCuda(float *mem, int start)
    {
        mem[start] = (float)x();
        mem[start + 1] = (float)z();
        mem[start + 2] = heading().rad();
        mem[start + 3] = speed_px_s();
        mem[start + 4] = _width_px;
        mem[start + 5] = _height_px;
        return start + 5;
    }


public:
    MovingObstacle(int x, int z, angle heading, float speed, int width_px, int height_px) : State(x, z, heading, speed)
    {
        _width_px = width_px;
        _height_px = height_px;
    }

    constexpr inline int width_px() { return _width_px; }
    constexpr inline int height_px() { return _height_px; }


    int copyToCuda(CudaPtr<float> mem, int start) {
        return copyToCuda(mem.get(), start);
    }
    int copyToCuda(std::shared_ptr<CudaPtr<float>> mem, int start) {
        return copyToCuda(mem->get(), start);
    }

    static std::shared_ptr<CudaPtr<float>> listToCuda(std::vector<MovingObstacle> values)
    {
        std::shared_ptr<CudaPtr<float>> mem = std::make_shared<CudaPtr<float>>(values.size());

        unsigned int s = 0;
        for (auto p : values)
        {
            s = p.copyToCuda(mem->get(), s);
        }

        return mem;
    }
};

constexpr inline int2 moving_obstacle_pos(float *mem, int pos)
{
    return {(int)mem[pos], (int)mem[pos + 1]};
}

constexpr inline float moving_obstacle_heading(float *mem, int pos)
{
    return mem[pos + 2];
}

constexpr inline float moving_obstacle_speed_px_s(float *mem, int pos)
{
    return mem[pos + 3];
}

constexpr inline int2 moving_obstacle_size(float *mem, int pos)
{
    return {(int)mem[pos + 4], (int)mem[pos + 5]};
}

#endif