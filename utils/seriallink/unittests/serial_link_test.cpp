#include <gtest/gtest.h>
#include <stdio.h>
#include <stdexcept>
#include "../serial_bus.h"
#include "../seriallink.h"
#include <thread>
#include <functional>
#include <chrono>

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
            writeByte(val[i]);
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
        readPos = 0;
        writePos = 0;
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

    void copyBuffer(SerialLinkTestBusSendReceive * out) {
        for (int i = 0; i < writePos; i++) {
            out->buffer[i] = buffer[i];
        }
        out->readPos = 0;
        out->writePos = writePos;
    }
};

class ThreadExecution
{
    SerialLink *_link;
    bool _keepExec;
    std::thread *_thr;
    std::function<void(SerialLink *)> method;

    void exec()
    {
        while (_keepExec)
        {
            method(this->_link);
        }
    }

public:
    ThreadExecution(SerialLink *link, std::function<void(SerialLink *)> method) : _keepExec(true), _link(link)
    {
        this->method = method;
        _keepExec = true;
        _thr = new std::thread(&ThreadExecution::exec, this);
    }

    ~ThreadExecution()
    {
        _keepExec = false;
        _thr->join();
        delete _thr;
    }
};

TEST(SerialLinkTest, TestSendReceive)
{
    SerialLinkTestBusSendReceive busSnd;
    SerialLinkTestBusSendReceive busRcv;
    SerialLink *linkSnd = new SerialLink(&busSnd, 8010);
    SerialLink *linkRcv = new SerialLink(&busRcv, 8010);

    auto execSnd = new ThreadExecution(linkSnd, [=](SerialLink *l)
                                       { l->loop(); });

    auto execRcv = new ThreadExecution(linkRcv, [=](SerialLink *l)
                                       { l->loop(); });

    while (!linkSnd->isReady())
        ;
    while (!linkRcv->isReady())
        ;

    delete execSnd;
    delete execRcv;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    busSnd.readPos = 0;
    busSnd.writePos = 0;
    busRcv.readPos = 0;
    busRcv.writePos = 0;

    int msg_size = 8000;

    char msg[msg_size];
    for (int i = 0; i < msg_size; i++)
        msg[i] = i % 254;

    linkSnd->write(msg, 8000);
    linkSnd->send();

    busSnd.copyBuffer(&busRcv);

    while (!linkRcv->hasData()) {
        linkRcv->loop();
    }

    char *rcvData = linkRcv->readString(8000);

    for (int i = 0; i < msg_size; i++)
        ASSERT_EQ(rcvData[i], msg[i]);

    delete rcvData;

    busRcv.writePos = 0;
    busRcv.readPos = 0;
    busSnd.writePos = 0;
    busSnd.readPos = 0;

    linkRcv->sendAck();
    busRcv.copyBuffer(&busSnd);

    ASSERT_TRUE(linkSnd->waitAck(true));

    delete linkSnd;
    delete linkRcv;
}