#include <string.h>
#include <stdio.h>

#include "serial_protocol.h"

#define RECV_PACKAGE_EMPTY 0
#define RECV_PACKAGE_OK 1
#define RECV_PACKAGE_PARTIAL 2
#define RECV_PACKAGE_NOK 3

#define min(x, y) x < y ? x : y

#define MAX_READ_SIZE 64

SerialLinkProtocol::SerialLinkProtocol(SerialBus *bus, int bufferSize) : _paramsNegotiated(false),
                                                                         _rcvBufferPos(0),
                                                                         _sndBufferPos(0),
                                                                         _readPos(0),
                                                                         _lastMessageAckState(MESSAGE_ACK_STATE_WAIT_RESP)
{
    this->bus = bus;
    this->rcvBuffer = new char[bufferSize];
    this->sndBuffer = new char[bufferSize];
    this->_bufferSize = bufferSize;
    this->bus->initialize();
    this->bus->rstTimeout();
    //__updateSendStartPos();
}
SerialLinkProtocol::~SerialLinkProtocol()
{
    delete[] this->rcvBuffer;
}

//
// SEND DATA
//
// void SerialLinkProtocol::__updateSendStartPos()
// {
//     _sndBufferPos = SIZE_INT + 2;
// }

void SerialLinkProtocol::writeByte(char data)
{
    this->sndBuffer[_sndBufferPos++] = data;
}
void SerialLinkProtocol::writeInt(int data)
{
    intp p;
    p.val = data;
    for (int i = 0; i < SIZE_INT; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeUInt(unsigned int data)
{
    uintp p;
    p.val = data;
    for (int i = 0; i < SIZE_UNSIGNED_INT; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeLong(long data)
{
    longp p;
    p.val = data;
    for (int i = 0; i < SIZE_LONG; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeULong(unsigned long data)
{
    ulongp p;
    p.val = data;
    for (int i = 0; i < SIZE_UNSIGNED_LONG; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeFloat(float data)
{
    floatp p;
    p.fval = data;
    for (int i = 0; i < SIZE_FLOAT; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeDouble(double data)
{
    doublep p;
    p.fval = data;
    for (int i = 0; i < SIZE_DOUBLE; i++)
        writeByte(p.bval[i]);
}
void SerialLinkProtocol::writeString(char *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        writeByte(data[i]);
    }
}
void SerialLinkProtocol::writeFloatArray(float *data, int size)
{
    floatp p;
    for (int i = 0; i < size; i++)
    {
        p.fval = data[i];
        for (int j = 0; j < SIZE_FLOAT; j++)
            writeByte(p.bval[j]);
    }
}
void SerialLinkProtocol::sendData()
{    
    this->bus->writeByte(MESSAGE_START);
    this->bus->writeByte(MESSAGE_TYPE_DATA);

    uintp p;
    p.val = _sndBufferPos;
    for (int i = 0; i < SIZE_UNSIGNED_INT; i++)
        this->bus->writeByte(p.bval[i]);

    this->bus->writeBytes(sndBuffer, _sndBufferPos);
    this->bus->writeByte(MESSAGE_END);
    _sndBufferPos = 0;
    _lastMessageAckState = MESSAGE_ACK_STATE_WAIT_RESP;
}

char SerialLinkProtocol::readByte()
{
    if (_readPos > this->_rcvBufferPos)
        return 0;
    return this->rcvBuffer[_readPos++];
}
int SerialLinkProtocol::readInt()
{
    if (_readPos > this->_rcvBufferPos - SIZE_INT)
        return 0;

    intp p;
    for (int i = 0; i < SIZE_INT; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_INT;
    return p.val;
}
unsigned int SerialLinkProtocol::readUInt()
{
    if (_readPos > this->_rcvBufferPos - SIZE_UNSIGNED_INT)
        return 0;

    uintp p;
    for (int i = 0; i < SIZE_UNSIGNED_INT; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_UNSIGNED_INT;
    return p.val;
}
long SerialLinkProtocol::readLong()
{
    if (_readPos > this->_rcvBufferPos - SIZE_LONG)
        return 0;

    longp p;
    for (int i = 0; i < SIZE_LONG; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_LONG;
    return p.val;
}
unsigned long SerialLinkProtocol::readULong()
{
    if (_readPos > this->_rcvBufferPos - SIZE_UNSIGNED_LONG)
        return 0;

    ulongp p;
    for (int i = 0; i < SIZE_LONG; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_UNSIGNED_LONG;
    return p.val;
}
float SerialLinkProtocol::readFloat()
{
    if (_readPos > this->_rcvBufferPos - SIZE_FLOAT)
        return 0;

    floatp p;
    for (int i = 0; i < SIZE_FLOAT; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_FLOAT;
    return p.fval;
}
double SerialLinkProtocol::readDouble()
{
    if (_readPos > this->_rcvBufferPos - SIZE_DOUBLE)
        return 0;

    doublep p;
    for (int i = 0; i < SIZE_DOUBLE; i++)
        p.bval[i] = this->rcvBuffer[_readPos + i];

    _readPos += SIZE_DOUBLE;
    return p.fval;
}
char *SerialLinkProtocol::readString(int size)
{
    if (_readPos > this->_rcvBufferPos - size)
        return nullptr;

    char *str = new char[size + 1];
    memcpy(str, this->rcvBuffer + _readPos, size);
    str[size] = 0;

    _readPos += size;
    return str;
}
float *SerialLinkProtocol::readFloat(int size)
{
    if (_readPos > this->_rcvBufferPos - (SIZE_FLOAT * size))
        return nullptr;

    float *arr = new float[size + 1];
    arr[size] = 0;

    floatp p;

    for (unsigned int i = 0; i < size; i++)
    {
        for (int j = 0; j < SIZE_FLOAT; j++)
        {
            p.bval[j] = readByte();
        }
        arr[i] = p.fval;
    }

    return arr;
}
void SerialLinkProtocol::flushRecvBuffer()
{
    this->_rcvBufferPos = 0;
    this->_readPos = 0;
}

void SerialLinkProtocol::receiveData()
{
    if (this->_rcvBufferPos > 0)
        return;

    if (bus->dataAvail() < 1)
        return;

    if (__readByteFromBuffer() != MESSAGE_START)
        return;

    switch (__readByteFromBuffer())
    {
    case MESSAGE_TYPE_DATA:
        __receiveMessage();
        break;

    case MESSAGE_TYPE_DATA_ACK:
        __receiveAck();
        break;

    case MESSAGE_TYPE_DATA_NACK:
        __receiveNack();
        break;

    default:
        // probably noise!
        break;
    }
}

void SerialLinkProtocol::__receiveMessage()
{
    unsigned int size = __readUnsignedIntFromBuffer();
    if (size <= 0)
        return;

    this->_rcvBufferPos = 0;
    this->_readPos = 0;

    if (size > _bufferSize)
        return;

    while (size > 0)
    {
        int readLen = min(size, MAX_READ_SIZE);
        __lockWaitForDataOnBus(readLen);
        auto readSize = bus->readBytesTo(rcvBuffer, _rcvBufferPos, readLen);
        size -= readSize;
        _rcvBufferPos += readSize;
    }

    if (__readByteFromBuffer() != MESSAGE_END)
    {
        this->_rcvBufferPos = 0;
        return;
    }
}
void SerialLinkProtocol::__receiveAck()
{
    if (__readByteFromBuffer() == MESSAGE_END)
        this->_lastMessageAckState = MESSAGE_ACK_STATE_ACK;
}
void SerialLinkProtocol::__receiveNack()
{
    if (__readByteFromBuffer() == MESSAGE_END)
        this->_lastMessageAckState = MESSAGE_ACK_STATE_NACK;
}

void SerialLinkProtocol::sendMessageAck()
{
    this->bus->writeByte(MESSAGE_START);
    this->bus->writeByte(MESSAGE_TYPE_DATA_ACK);
    this->bus->writeByte(MESSAGE_END);
}
void SerialLinkProtocol::sendMessageNack()
{
    this->bus->writeByte(MESSAGE_START);
    this->bus->writeByte(MESSAGE_TYPE_DATA_NACK);
    this->bus->writeByte(MESSAGE_END);
}

unsigned int SerialLinkProtocol::__readUnsignedIntFromBuffer()
{
    __lockWaitForDataOnBus(SIZE_UNSIGNED_INT);
    intp p;
    p.val = 0;
    for (int i = 0; i < SIZE_INT; i++)
    {
        p.bval[i] = this->bus->readByte();
    }
    return p.val;
}
char SerialLinkProtocol::__readByteFromBuffer()
{
    __lockWaitForDataOnBus(1);
    return this->bus->readByte();
}

void SerialLinkProtocol::__sendDataSize(int size)
{
    intp len;
    len.val = size;
    for (int i = 0; i < SIZE_INT; i++)
    {
        this->bus->writeByte(len.bval[i]);
    }
}

void SerialLinkProtocol::__lockWaitForDataOnBus(int min)
{
    while (bus->dataAvail() < min)
    {
        bus->waitBus();
    }
}

bool SerialLinkProtocol::hasData()
{
    return _rcvBufferPos > 0;
}

unsigned int SerialLinkProtocol::rcvSize()
{
    return this->_rcvBufferPos;
}

bool SerialLinkProtocol::hasNegotiatedParams()
{
    return this->_paramsNegotiated;
}

bool SerialLinkProtocol::checkTimeout()
{
    return this->bus->checkTimeout();
}

bool SerialLinkProtocol::busReady()
{
    return this->bus->isReady();
}

bool SerialLinkProtocol::checkMessageAck()
{
    return _lastMessageAckState == MESSAGE_ACK_STATE_ACK;
}
bool SerialLinkProtocol::checkMessageNack()
{
    return _lastMessageAckState == MESSAGE_ACK_STATE_NACK;
}
