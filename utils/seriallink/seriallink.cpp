#include "seriallink.h"

SerialLink::SerialLink(SerialBus *bus, int bufferSize)
{
    _protocol = new SerialLinkProtocol(bus, bufferSize);
}
SerialLink::~SerialLink()
{
    if (_protocol != nullptr)
        delete _protocol;

    _protocol = nullptr;
}

bool SerialLink::isReady()
{
    return this->_protocol->busReady();
}

bool SerialLink::hasData()
{
    return _protocol->hasData();
}


unsigned int SerialLink::dataSize()
{
    return _protocol->rcvSize();
}

void SerialLink::write(char *data, long dataSize)
{
    _protocol->writeString(data, dataSize);
}
void SerialLink::write(float *data, long dataSize)
{
    _protocol->writeFloatArray(data, dataSize);
}
void SerialLink::writeByte(char data)
{
    _protocol->writeByte(data);
}
void SerialLink::write(int data)
{
    _protocol->writeInt(data);
}
void SerialLink::write(unsigned int data)
{
    _protocol->writeUInt(data);
}
void SerialLink::write(long data)
{
    _protocol->writeLong(data);
}
void SerialLink::write(unsigned long data)
{
    _protocol->writeULong(data);
}
void SerialLink::write(float data)
{
    _protocol->writeFloat(data);
}
void SerialLink::write(double data)
{
    _protocol->writeDouble(data);
}
void SerialLink::send()
{
    _protocol->sendData();
}

char SerialLink::readByte()
{
    return _protocol->readByte();
}
int SerialLink::readInt()
{
    return _protocol->readInt();
}
unsigned int SerialLink::readUInt()
{
    return _protocol->readInt();
}
long SerialLink::readLong()
{
    return _protocol->readLong();
}
unsigned long SerialLink::readULong()
{
    return _protocol->readULong();
}
float SerialLink::readFloat()
{
    return _protocol->readFloat();
}
double SerialLink::readDouble()
{
    return _protocol->readDouble();
}
char *SerialLink::readString(int size)
{
    return _protocol->readString(size);
}
float *SerialLink::readFloat(int size)
{
    return _protocol->readFloat(size);
}
void SerialLink::endRead()
{
    _protocol->flushRecvBuffer();
}
void SerialLink::sendAck()
{
    _protocol->sendMessageAck();
}
bool SerialLink::waitAck(bool executeLoop)
{
    while (!_protocol->checkTimeout())
    {
        if (executeLoop)
            loop();

        if (_protocol->checkMessageAck())
            return true;
            
        if (_protocol->checkMessageNack())
            return false;
    }
    return false;
}


void SerialLink::loop()
{
    if (!_protocol->busReady())
        return;

    _protocol->receiveData();
}