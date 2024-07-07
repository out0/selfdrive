#include "datalink_connection.h"
#include <iostream>
#include <sys/socket.h> // Include for socket APIs
#include <netinet/in.h> // Include for internet protocols
#include <cstring>      // Include for string operations
#include <unistd.h>     // Include for POSIX operating system API
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>

#define DATALINK_MRU (long)1024
#define DATALINK_MTU (long)1024

DatalinkConnection::DatalinkConnection(int noDataTimeout_ms)
{
    this->_isConnected = false;
    this->timeoutStart = -1;
    this->noDataTimeout_ms = noDataTimeout_ms;
    if (noDataTimeout_ms <= 0)
        this->noDataHalfTimeout_ms = -1;
    else
        this->noDataHalfTimeout_ms = std::floor(this->noDataTimeout_ms / 2);

    memset(fds, 0, sizeof(fds));
}

long long DatalinkConnection::timeNow()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
void DatalinkConnection::rstTimeout()
{
    timeoutStart = timeNow();
}

void DatalinkConnection::setConnected(bool val)
{
    this->_isConnected = val;
}

bool DatalinkConnection::checkTimeout()
{
    if (this->noDataTimeout_ms <= 0)
        return false;

    if (timeoutStart <= 0)
    {
        timeoutStart = timeNow();
        return false;
    }

    if (timeNow() - timeoutStart > noDataTimeout_ms)
    {
#ifdef DEBUG
        printf("[datalink] TIMEOUT\n");
#endif
        return true;
    }
    return false;
}

bool DatalinkConnection::isConnected()
{
    return this->_isConnected.load();
}

bool DatalinkConnection::checkConnectionLost(int errorCode)
{
    switch (errorCode)
    {
    case 9:
    case 32:  // broken pipe
    case 50:  // network is down
    case 51:  // network unreachable
    case 52:  // network dropped connection
    case 53:  // aborted
    case 54:  // connection reset by peer
    case 104: // connection reset by peer
#ifdef DEBUG
        printf("[datalink] connection lost code: %d\n", errorCode);
        perror("reason\n");
#endif
        return true;
    case EWOULDBLOCK: // 11
        return false;
    default:
        printf("[datalink] unknown error code: %d\n", errorCode);
        perror("reason\n");
        return true;
    }
}

bool DatalinkConnection::socketReadableCheck(int sock, int timeout_ms)
{
    fds[0].fd = sock;
    fds[0].events = POLLIN;

    int ret = poll(fds, 1, timeout_ms);
    if (ret == -1)
    {
        std::cerr << "[datalink] poll() failed" << std::endl;
        return false;
    }
    if (ret == 0) // timeout reached
        return false;

    return fds[0].revents & POLLIN;
}

bool DatalinkConnection::readFromSocket(int socket, char *buffer, long expectedSize)
{
    if (!this->_isConnected)
        return false;

    long totalSize = 0;
    long maxBytesToRead = -1;

    while (totalSize < expectedSize)
    {
        if (checkTimeout())
            return false;

        maxBytesToRead = std::min(expectedSize - totalSize, DATALINK_MRU);

        if (!socketReadableCheck(socket, noDataTimeout_ms))
            return false;

        long partialSize = recv(socket, buffer + totalSize, maxBytesToRead, 0);

        if (partialSize < 0)
        {
            if (checkConnectionLost(errno))
                return false;

            // a recovable error
            continue;
        }

        if (partialSize == 0)
            return false; // connection reset by peer

        totalSize += partialSize;
        rstTimeout();
    }

    return true;
}

bool DatalinkConnection::writeToSocket(int socket, char *buffer, long expectedSize)
{
    if (!this->_isConnected)
        return false;

    long totalSize = 0;
    long maxBytesToWrite = -1;

    while (totalSize < expectedSize)
    {
        if (checkTimeout()) 
            return false;
        

        maxBytesToWrite = std::min(expectedSize - totalSize, DATALINK_MTU);

        long partialSize = send(socket, buffer + totalSize, maxBytesToWrite, MSG_NOSIGNAL);

        if (partialSize < 0)
        {
            if (checkConnectionLost(errno))
                return false;
            
        }

        totalSize += partialSize;
        rstTimeout();
    }

    return true;
}

void DatalinkConnection::closeSocket(int sock)
{
    if (isConnected())
        close(sock);
    setConnected(false);
}