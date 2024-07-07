#ifndef H_SERIAL_LINK
#define H_SERIAL_LINK

#include <serial_bus.h>
#include <serial_protocol.h>


class SerialLink
{

private:
    SerialLinkProtocol * _protocol; 

public:
    SerialLink(SerialBus *bus, int bufferSize);
    ~SerialLink();

    bool isReady();
    bool hasData();
    unsigned int dataSize();

    void write(char *data, long dataSize);
    void write(float *data, long dataSize);
    void writeByte(char data);
    void write(int data);
    void write(unsigned int data);
    void write(long data);
    void write(unsigned long data);
    void write(float data);
    void write(double data);
    void send();

    char readByte();
    int readInt();
    unsigned int readUInt();
    long readLong();
    unsigned long readULong();
    float readFloat();
    double readDouble();
    char * readString(int size);
    float * readFloat(int size);
    void endRead();

    void loop();

    void sendAck();
    bool waitAck(bool executeLoop);
};

#endif