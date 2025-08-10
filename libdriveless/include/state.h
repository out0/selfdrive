#pragma once

#ifndef __STATE_DRIVELESS_H
#define __STATE_DRIVELESS_H

#include "angle.h"
#include <iostream>

class State {
    int _x;
    int _z;
    angle _heading;
    float _speed_px_s;

public:
    State(int x, int z, angle heading, float speed) : _x(x), _z(z), _heading(heading), _speed_px_s(speed) {}

    int x() { return _x; }
    int z() { return _z; }
    float speed_px_s() { return _speed_px_s; }
    angle heading() { return _heading; }  

    inline bool operator==(const State &other)
    {
        return _x == other._x &&
               _z == other._z &&
               _heading == other._heading;
    }
    inline bool operator!=(const State &other)
    {
        return !(*this == other);
    }
    void set_heading(float new_heading_rad) {
        _heading.setRad(new_heading_rad);
    }

};



#endif