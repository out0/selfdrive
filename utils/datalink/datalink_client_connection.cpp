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

#include "datalink_client_connection.h"

DatalinkClientConnection::DatalinkClientConnection(int noDataTimeout_ms) : DatalinkConnection(noDataTimeout_ms)
{
}
DatalinkClientConnection::~DatalinkClientConnection()
{
    closeConnection();
}

bool DatalinkClientConnection::openConnection(const char *host, int port)
{
    setConnected(false);

    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    this->connSock = socket(AF_INET, SOCK_STREAM, 0);
    if (this->connSock < 0)
    {
        perror("[datalink] socket");
        return false;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &server_addr.sin_addr) <= 0) {
        printf("[datalink] host %s is invalid\n", host);
        return false;
    }

    // Connect to the server
    if (connect(this->connSock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        close(this->connSock);
        printf("[datalink] failed to connect to %s, %d\n", host, port);
        perror("[datalink] connect");
        return false;
    }

#ifdef DEBUG
    printf("[datalink] connected\n");
#endif

    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(this->connSock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));

    rstTimeout();
    setConnected(true);
    return true;
}

void DatalinkClientConnection::closeConnection() {
   closeSocket(this->connSock);
}

bool DatalinkClientConnection::readFromSocket(char *buffer, long expectedSize)
{
    return DatalinkConnection::readFromSocket(this->connSock, buffer, expectedSize);
}
bool DatalinkClientConnection::writeToSocket(char *buffer, long expectedSize)
{
    return DatalinkConnection::writeToSocket(this->connSock, buffer, expectedSize);
}
