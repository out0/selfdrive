
#ifndef _SERIAL_BUS_H
#define _SERIAL_BUS_H

class SerialBus
{

public:
    virtual bool initialize() = 0;

    virtual int dataAvail() = 0;

    virtual unsigned char readByte() = 0;

    virtual unsigned int readBytesTo(char *buffer, int start, int maxSize) = 0;

    virtual void writeByte(unsigned char val) = 0;

    virtual void writeBytes(char *buffer, int size) = 0;

    virtual void flush() = 0;

    virtual bool isReady() = 0;

    virtual void waitBus() = 0;

    virtual void rstTimeout() = 0;

    virtual bool checkTimeout() = 0;
};

#endif
