#ifndef __DATALINK_SERVER_CONNECTION_H
#define __DATALINK_SERVER_CONNECTION_H

#include "datalink_connection.h"
#include <netinet/in.h> // Include for internet protocols

class DatalinkServerConnection : public DatalinkConnection
{
private:
    struct sockaddr_in listenerAddress;
    int listenSock, connSock;
    bool validListenSock;

public:
    DatalinkServerConnection(int noDataTimeout_ms);
    ~DatalinkServerConnection();

    bool bindConnection(int port);
    bool socketAcceptIncommingConnections();
    void closeListen();
    void closeConnection();
    bool readFromSocket(char *buffer, long expectedSize);
    bool writeToSocket(char *buffer, long expectedSize);
};

#endif