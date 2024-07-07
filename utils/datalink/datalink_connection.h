#ifndef __DATALINK_CONNECTION_H
#define __DATALINK_CONNECTION_H
#include <sys/poll.h>
#include <atomic>


class DatalinkConnection
{
private:
    long long timeoutStart;
    int noDataTimeout_ms;
    int noDataHalfTimeout_ms;
    std::atomic<bool> _isConnected;
    struct pollfd fds[1];

protected:    
    DatalinkConnection(int noDataTimeout_ms);
    void setConnected(bool val);
    bool checkConnectionLost(int errorCode);
    bool socketReadableCheck(int sock, int timeout_ms);
    bool readFromSocket(int socket, char *buffer, long expectedSize);
    bool writeToSocket(int socket, char *buffer, long expectedSize);
    void closeSocket(int sock);

public:
    static long long timeNow();
    void rstTimeout();
    bool checkTimeout();
    bool isConnected();
    virtual ~DatalinkConnection() = default;
};

#endif