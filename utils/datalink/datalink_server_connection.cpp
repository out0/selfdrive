#include "datalink.h"
#include <iostream>
#include <sys/socket.h> // Include for socket APIs
#include <netinet/in.h> // Include for internet protocols
#include <cstring>      // Include for string operations
#include <unistd.h>     // Include for POSIX operating system API
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <string.h>
#include <algorithm>
#include <poll.h>

#include "datalink_server_connection.h"

#define TIMEOUT_ACCEPT_ms 100
// #define DEBUG 1

DatalinkServerConnection::DatalinkServerConnection(int noDataTimeout_ms) : DatalinkConnection(noDataTimeout_ms)
{
    this->validListenSock = false;
}
DatalinkServerConnection::~DatalinkServerConnection()
{
    closeConnection();
}

bool DatalinkServerConnection::bindConnection(int port)
{
    this->validListenSock = false;
    setConnected(false);

    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    listenSock = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSock < 0)
    {
        perror("[datalink] socket");
        return false;
    }

    if (setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
    {
        perror("[datalink] setsockopt"); // Print error if setting socket options fails
        return false;
    }

    address.sin_family = AF_INET;         // Set address family to AF_INET (IPv4)
    address.sin_addr.s_addr = INADDR_ANY; // Accept connections from any IP
    address.sin_port = htons(port);       // Set port to 8080 with proper byte order

    int on = 1;
    if (ioctl(listenSock, FIONBIO, (char *)&on) < 0)
    {
        perror("[datalink] ioctl() failed");
        close(listenSock);
        return false;
    }

    if (bind(listenSock, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("[datalink] bind failed");
        close(listenSock);
        return false;
    }

    if (listen(listenSock, 1) < 0)
        return false;

    this->validListenSock = true;

    return true;
}

bool DatalinkServerConnection::socketAcceptIncommingConnections()
{
    setConnected(false);

#ifdef DEBUG
    printf("[datalink] listening for incomming connections...\n");
#endif

    if (!socketReadableCheck(listenSock, TIMEOUT_ACCEPT_ms))
        return false;

    // struct sockaddr_in address;
    // int addrlen = sizeof(address);

    // if ((connSock = accept(this->listenSock, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
    connSock = accept(this->listenSock, nullptr, nullptr);

    if (connSock < 0)
    {
        perror("[datalink] accept");
        return false;
    }

    rstTimeout();
    setConnected(true);
    return true;
}

void DatalinkServerConnection::closeListen()
{
    if (!this->validListenSock)
        return;

    close(this->listenSock);
    this->validListenSock = false;
}

void DatalinkServerConnection::closeConnection()
{
    if (!isConnected())
        return;

    close(this->connSock);
    setConnected(false);
}

bool DatalinkServerConnection::readFromSocket(char *buffer, long expectedSize)
{
    return DatalinkConnection::readFromSocket(this->connSock, buffer, expectedSize);
}
bool DatalinkServerConnection::writeToSocket(char *buffer, long expectedSize)
{
    return DatalinkConnection::writeToSocket(this->connSock, buffer, expectedSize);
}