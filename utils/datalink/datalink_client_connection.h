#ifndef __DATALINK_CLIENT_CONNECTION_H
#define __DATALINK_CLIENT_CONNECTION_H

#include "datalink_connection.h"
#include <netinet/in.h> // Include for internet protocols

class DatalinkClientConnection : public DatalinkConnection
{
private:
    int connSock;
    struct sockaddr_in server_addr;

public:
    DatalinkClientConnection(int noDataTimeout_ms);
    ~DatalinkClientConnection();

    bool openConnection(const char *host, int port);
    void closeConnection();
    bool readFromSocket(char *buffer, long expectedSize);
    bool writeToSocket(char *buffer, long expectedSize);
};

#endif
