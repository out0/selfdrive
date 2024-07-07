#include <gtest/gtest.h>
#include <stdio.h>
#include <stdexcept>
#include "../serial_protocol.h"
#include "../serial_bus.h"
#include "../serial_protocol.h"

class SerialLinkTestBusSend : public SerialBus
{
public:
    SerialLinkTestBusSend() : SerialBus(),
                              initialize_called(false),
                              sendPos(0)
    {
    }

    bool initialize_called;
    char sendBuffer[8010];
    int sendPos;

    void initialize()
    {
        initialize_called = true;
    }

    int dataAvail()
    {
        return 0;
    }

    unsigned char readByte() override
    {
        throw std::domain_error("readByte() shouldn've be called");
    }

    unsigned int readBytesTo(char *buffer, int start, int maxSize) override
    {
        throw std::domain_error("readBytesTo() shouldn've be called");
    }

    void writeByte(unsigned char val) override
    {
        sendBuffer[sendPos++] = val;
    }
    void writeBytes(char *val, int size) override
    {
        for (int i = 0; i < size; i++)
            sendBuffer[sendPos++] = val[i];
    }

    void flush() override
    {
        throw std::domain_error("flush() shouldn've be called");
    }

    bool isReady() override
    {
        return true;
    }

    void waitBus() override
    {
    }

    void rstTimeout() override
    {
    }

    bool checkTimeout() override
    {
        return false;
    }
};

TEST(SerialProtocolTest, TestSend)
{
    SerialLinkTestBusSend bus;
    SerialLinkProtocol link(&bus, 8010);

    int msg_size = 8000;

    char msg[msg_size];
    for (int i = 0; i < msg_size; i++)
        msg[i] = i % 254;

    link.writeString(msg, msg_size);
    link.sendData();

    ASSERT_EQ(bus.sendPos, msg_size + SIZE_UNSIGNED_INT + 3);

    ASSERT_EQ(bus.sendBuffer[0], MESSAGE_START);
    ASSERT_EQ(bus.sendBuffer[1], MESSAGE_TYPE_DATA);

    uintp p;
    p.val = msg_size;

    for (int i = 0; i < SIZE_UNSIGNED_INT; i++)
        ASSERT_EQ(bus.sendBuffer[i + 2], p.bval[i]);

    for (int i = 0; i < msg_size; i++)
        ASSERT_EQ(bus.sendBuffer[SIZE_UNSIGNED_INT + 2 + i], msg[i]);

    ASSERT_EQ(bus.sendBuffer[msg_size + SIZE_UNSIGNED_INT + 2], MESSAGE_END);
}

class SerialLinkTestBusSendReceive : public SerialBus
{
public:
    SerialLinkTestBusSendReceive() : SerialBus(),
                                     initialize_called(false),
                                     writePos(0),
                                     readPos(0)
    {
    }

    bool initialize_called;
    char buffer[8010];
    int writePos;
    int readPos;

    void initialize()
    {
        initialize_called = true;
    }

    int dataAvail() override
    {
        return writePos - readPos;
    }

    unsigned char readByte() override
    {
        if (readPos >= writePos)
            return -1;
        return buffer[readPos++];
    }

    void writeByte(unsigned char val) override
    {
        buffer[writePos++] = val;
    }

    void writeBytes(char *val, int size) override
    {
        for (int i = 0; i < size; i++)
            buffer[writePos++] = val[i];
    }

    unsigned int readBytesTo(char *buf, int start, int maxSize) override
    {
        if (readPos >= writePos)
            return -1;

        for (int i = 0; i < maxSize; i++)
            buf[i + start] = buffer[readPos++];

        return maxSize;
    }

    void flush() override
    {
        throw std::domain_error("flush() shouldn've be called");
    }

    bool isReady() override
    {
        return true;
    }

    void waitBus() override
    {
    }

    void rstTimeout() override
    {
    }

    bool checkTimeout() override
    {
        return false;
    }
};

TEST(SerialProtocolTest, TestReceive)
{
    SerialLinkTestBusSendReceive bus;
    SerialLinkProtocol link(&bus, 8010);

    int msg_size = 8000;

    char msg[msg_size];
    for (int i = 0; i < msg_size; i++)
        msg[i] = i % 254;

    link.writeString(msg, msg_size);
    link.sendData();

    link.receiveData();

    ASSERT_EQ(link.rcvSize(), msg_size);

    for (int i = 0; i < link.rcvSize(); i++)
    {
        ASSERT_EQ(link.readByte(), msg[i]);
    }

}

TEST(SerialProtocolTest, TestAck)
{
    SerialLinkTestBusSendReceive bus;
    SerialLinkProtocol link(&bus, 10);
    memset(bus.buffer, 0, 1000);
    int i = 0;
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);

    link.sendMessageAck();
    i = 0;
    ASSERT_EQ(bus.buffer[i++], MESSAGE_START);
    ASSERT_EQ(bus.buffer[i++], MESSAGE_TYPE_DATA_ACK);
    ASSERT_EQ(bus.buffer[i++], MESSAGE_END);
    ASSERT_EQ(bus.buffer[i++], 0);
}

TEST(SerialProtocolTest, TestNack)
{
    SerialLinkTestBusSendReceive bus;
    SerialLinkProtocol link(&bus, 10);
    memset(bus.buffer, 0, 1000);
    int i = 0;
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);
    ASSERT_EQ(bus.buffer[i++], 0);

    link.sendMessageNack();
    i = 0;
    ASSERT_EQ(bus.buffer[i++], MESSAGE_START);
    ASSERT_EQ(bus.buffer[i++], MESSAGE_TYPE_DATA_NACK);
    ASSERT_EQ(bus.buffer[i++], MESSAGE_END);
    ASSERT_EQ(bus.buffer[i++], 0);
}