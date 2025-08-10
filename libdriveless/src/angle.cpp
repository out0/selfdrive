#include "../include/angle.h"

angle angle::deg(double val)
{
    return angle(TO_RAD * val);
}

angle angle::rad(double val)
{
    return angle(val);
}

void angle::setDeg(double val)
{
    _val = TO_RAD * val;
}

void angle::setRad(double val)
{
    _val = val;
}

