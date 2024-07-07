#include <string>
#include <memory>
#include <cstring>
#include "ego_car.h"

extern "C"
{
    void *initialize(const char *device)
    {
        auto ego = new EgoCar();
        if (ego->initialize(device))
            return ego;
        
        delete ego;
        return nullptr;
    }
    bool resetActuators(void *ego)
    {
        if (ego == nullptr) return false;
        return ((EgoCar *)ego)->resetActuators();
    }
    bool setPower(void *ego, float power)
    {
        if (ego == nullptr) return false;
        return ((EgoCar *)ego)->setPower(power);
    }
    bool stop(void *ego)
    {
        if (ego == nullptr) return false;
        return ((EgoCar *)ego)->stop();
    }
    bool setSteeringAngle(void *ego, float angle)
    {
        if (ego == nullptr) return false;
        return ((EgoCar *)ego)->setSteeringAngle(angle);
    }
}
