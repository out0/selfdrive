#ifndef __EGO_CAR_H
#define __EGO_CAR_H

#include <memory>
#include <seriallink.h>
#include <serial_bus_linux.h>
#include <thread>

class EgoCar
{
private:
    std::unique_ptr<SerialBus> bus;
    std::unique_ptr<SerialLink> link;
    std::unique_ptr<std::thread> rcvThread;
    void __receiveData();
    void __debugShowCmdEcho();
    bool run;

public:
    EgoCar();
    ~EgoCar();
    bool initialize(const char *device);
    bool resetActuators();
    bool setPower(float power);
    bool stop();
    bool setSteeringAngle(float angle);
};

#endif