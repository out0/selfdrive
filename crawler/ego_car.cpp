#include "ego_car.h"
#include <fstream>
#include <sys/stat.h>
#include <serial_bus_linux.h>

#include "arduino_controller/crawler_protocol.h"
#define TIMEOUT 100
//#define TIMEOUT -1

static inline bool fileExists(const char *name)
{
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

bool EgoCar::initialize(const char *device)
{
    if (!fileExists(device))
        return false;

    bus = std::make_unique<SerialBusLinux>(device, TIMEOUT);
    if (!bus->initialize())
        return false;

    while (!bus->isReady())
        ;

    link = std::make_unique<SerialLink>(bus.get(), 128);

    rcvThread = std::make_unique<std::thread>(&EgoCar::__receiveData, this);

    return true;
}
bool EgoCar::resetActuators()
{
    link->writeByte(CMD_RST_ACTUATORS);
    link->send();
    return link->waitAck(false);
}
bool EgoCar::setPower(float power)
{
    link->writeByte(CMD_SET_POWER);
    link->write(power);
    link->send();
    return link->waitAck(false);
}
bool EgoCar::stop()
{
    link->writeByte(CMD_STOP);
    link->send();
    return link->waitAck(false);
}
bool EgoCar::setSteeringAngle(float angle)
{
    link->writeByte(CMD_SET_STEERING_ANGLE);
    link->write(angle);
    link->send();
    return link->waitAck(false);
}

void EgoCar::__receiveData()
{
    run = true;
    while (run)
    {
        link->loop();

        if (link->hasData())
        {
            unsigned char dataType = link->readByte();

            switch (dataType)
            {
            case DATA_CMD_ECHO:
                __debugShowCmdEcho();
                break;
            default:
                break;
            }

            link->endRead();
        }

        bus->waitBus();
    }
}

void EgoCar::__debugShowCmdEcho()
{
    unsigned char cmd = link->readByte();
    switch (cmd)
    {
    case CMD_RST_ACTUATORS:
        fprintf(stdout, "Received Command: RESET ACTUATORS\n");
        break;
    case CMD_SET_POWER:
        fprintf(stdout, "Received Command: SET POWER (%f)\n", link->readFloat());
        break;
    case CMD_STOP:
        fprintf(stdout, "Received Command: STOP\n");
        break;
    case CMD_SET_STEERING_ANGLE:
        fprintf(stdout, "Received Command: SET STEERING ANGLE (%f)\n", link->readFloat());
        break;
    default:
        break;
    }
}

EgoCar::EgoCar() : run(false)
{
}

EgoCar::~EgoCar()
{
    run = false;
    rcvThread->join();
}