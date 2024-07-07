#ifndef H_SERIAL_PROTOCOL
#define H_SERIAL_PROTOCOL

#define MESSAGE_START 0x0F
#define MESSAGE_TYPE_DATA 0x01
#define MESSAGE_TYPE_DATA_ACK 0x09
#define MESSAGE_TYPE_DATA_NACK 0x05
#define MESSAGE_END 0x0C

#define MESSAGE_ACK_STATE_WAIT_RESP 0
#define MESSAGE_ACK_STATE_ACK 1
#define MESSAGE_ACK_STATE_NACK 2

#define SIZE_INT 2
#define SIZE_UNSIGNED_INT 2
#define SIZE_LONG 4
#define SIZE_UNSIGNED_LONG 4
#define SIZE_FLOAT 4
#define SIZE_DOUBLE 4

typedef union
{
    float fval;
    unsigned char bval[SIZE_FLOAT];
} floatp;

typedef union
{
    double fval;
    unsigned char bval[SIZE_DOUBLE];
} doublep;

typedef union
{
    unsigned int val;
    unsigned char bval[SIZE_UNSIGNED_INT];
} uintp;

typedef union
{
    int val;
    unsigned char bval[SIZE_INT];
} intp;

typedef union
{
    unsigned long val;
    unsigned char bval[SIZE_UNSIGNED_LONG];
} ulongp;

typedef union
{
    long val;
    unsigned char bval[SIZE_LONG];
} longp;

#include "serial_bus.h"

class SerialRawData
{
public:    
    char *raw_data;
    int size;

    SerialRawData(int size) {
        this->raw_data = new char[size + 1];
        this->size = size;
    }

    // ~SerialRawData() {
    //     delete []raw_data;
    // }

};

class SerialLinkProtocol
{
    SerialBus *bus;
    char *rcvBuffer;
    char *sndBuffer;
    unsigned int _rcvBufferPos;
    unsigned int _sndBufferPos;
    unsigned int _readPos;
    int _bufferSize;

    bool _paramsNegotiated;
    int _lastMessageAckState;

private:

    
    void __sendDataSize(int size);
    void __lockWaitForDataOnBus(int min);

    char __readByteFromBuffer();
    unsigned int __readUnsignedIntFromBuffer();

    void __receiveMessage();
    void __receiveAck();
    void __receiveNack();


public:
    SerialLinkProtocol(SerialBus *bus, int bufferSize);
    ~SerialLinkProtocol();

    bool hasData();
    unsigned int rcvSize();
    
    void sendMessageAck();
    void sendMessageNack();
    bool checkMessageAck();
    bool checkMessageNack();


    void writeByte(char data);
    void writeInt(int data);
    void writeUInt(unsigned int data);
    void writeLong(long data);
    void writeULong(unsigned long data);
    void writeFloat(float data);
    void writeDouble(double data);
    void writeString(char *data, int size);
    void writeFloatArray(float *data, int size);
    void sendData();
    
    char readByte();
    int readInt();
    unsigned int readUInt();
    long readLong();
    unsigned long readULong();
    float readFloat();
    double readDouble();
    char * readString(int size);
    float * readFloat(int size);
    void receiveData();


    void flushRecvBuffer();

    // ---------

    bool hasNegotiatedParams();
    bool checkTimeout();
    bool busReady();
};


#endif