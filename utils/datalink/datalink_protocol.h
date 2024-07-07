#ifndef __DATALINK_PROTOCOL_H
#define __DATALINK_PROTOCOL_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <atomic>

#include "datalink_connection.h"
#include "datalink_client_connection.h"
#include "datalink_server_connection.h"
#include "datalink_result.h"
#define PROTOCOL_START_MESSAGE_SIZE (10 + sizeof(long)) * sizeof(char)
#define PROTOCOL_END_MESSAGE_SIZE 10 * sizeof(char)



class DatalinkProtocol
{
private:
    char *host;
    int port;
    std::unique_ptr<DatalinkConnection> conn;
    char messageStart[PROTOCOL_START_MESSAGE_SIZE + sizeof(char)];
    const char messageEnd[PROTOCOL_END_MESSAGE_SIZE + sizeof(char)] = {0x22, 0x22, 0x22, 0x22, 0x22, 0x23, 0x23, 0x23, 0x23, 0x23, 0x00};
    std::unique_ptr<DataLinkResult<char>> lastBuffer;
    std::unique_ptr<char[]> protocolBufferRcv;
    std::atomic<bool> bufferOnUpdate{false};
    std::atomic<bool> requestLockBufferUpdate{false};
    bool _isServer;

    bool readFromSocket(char **buffer, long size);
    bool writeToSocket(char *buffer, long size);
    bool checkRepeatByte(char *buffer, int startPos, int count, int byte);
    bool updateBuffer(long size);
    long checkMessageStart();
    bool checkMessageEnd();
    bool writeMessageStart(long messageSize);
    bool writeMessageEnd();
    void initialize();

public:
    DatalinkProtocol(const char *host, int port, int timeout_ms);
    DatalinkProtocol(int port, int timeout_ms);
    ~DatalinkProtocol();
    bool readMessage();
    bool sendPingMessage();
    bool writeMessage(char *data, long size);
    bool writeMessage(float *data, long size);
    bool writeMessage(uint8_t *data, long dataSize);
    std::shared_ptr<DataLinkResult<char>> releaseBuffer();
    bool hasData();
    bool isServer();
    bool bindConnection();
    bool openConnection();
    bool socketAcceptIncommingConnections();
    bool checkTimeout();
    void closeDataConnection();
    void closeListen();
};

typedef union
{
    float fval;
    char bval[sizeof(float)];
} floatp;

typedef union
{
    long val;
    char bval[sizeof(long)];
} longp;

typedef union uint8
{
    uint8_t val;
    char bval;
} uint8;

#endif