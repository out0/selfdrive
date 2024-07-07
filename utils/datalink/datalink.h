#ifndef __DATALINK_H
#define __DATALINK_H

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

#include "datalink_result.h"

#ifdef __DATALINK_COMPILE
#include "datalink_protocol.h"

enum DatalinkState
{
    CLOSED = 0,
    BIND_PORT = 1,
    LISTEN = 2,
    CONNECT = 3,
    RECEIVE_DATA = 4,
    TERMINATE_SESSION = 5
};

#endif


class Datalink
{
private:
#ifdef __DATALINK_COMPILE
    std::unique_ptr<DatalinkProtocol> linkProtocol;
    std::thread *asyncExecutor;
    std::thread *asyncPingKeepAlive;
    std::atomic<bool> run{false};
    std::atomic<int> state{CLOSED};
    std::mutex sendMtx;
    int halfTimeout_ms;
    void asyncExec();
    int bindPortExec();
    int listenExec();
    int receiveDataExec();
    int connectExec();
    int terminateSessionExec();
    void pingKeepAliveExec();
    void sendPingData();
#endif
protected:
    virtual void onDataReceived() {
        
    }

public:
    Datalink(const char *host, int port, int timeout_ms = 500);
    Datalink(int port, int timeout_ms = 500);
    ~Datalink();
    void destroy();
    bool isReady();
    bool waitReady(int timeout_ms);
    bool hasData();

    bool sendData(char *data, long data_size);
    bool sendData(float *data, long data_size);
    bool sendData(uint8_t *data, long dataSize);

    std::shared_ptr<DataLinkResult<char>> receiveData(int timeout_ms = -1);
    std::shared_ptr<DataLinkResult<float>> receiveDataF(int timeout_ms = -1);
};

#endif