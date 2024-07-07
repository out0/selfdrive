#ifndef H_SERIAL_BUS
#define H_SERIAL_BUS

#include "serial_bus.h"

class SerialBusLinux : public SerialBus
{
private:
    const char *device;
    int linkFd;
    long long timeoutStart;
    int noDataTimeout_ms;

public:
    SerialBusLinux(const char *device, int noDataTimeout_ms);

    bool initialize() override;

    int dataAvail() override;

    unsigned char readByte() override;

    unsigned int readBytesTo(char *buffer, int start, int maxSize) override;

    void writeByte(unsigned char val) override;

    void writeBytes(char *buffer, int size) override;

    void flush() override;

    bool isReady() override;

    void waitBus() override;

    void rstTimeout() override;

    bool checkTimeout() override;
};

#endif
