#include <cstring>

#include "datalink_protocol.h"

// #define DEBUG 1

void printBuffer(char *buffer, long size)
{
    printf("[");
    for (long i = 0; i < size; i++)
        printf(" %d", buffer[i]);
    printf(" ]\n");
}

bool DatalinkProtocol::checkRepeatByte(char *buffer, int startPos, int count, int byte)
{
    for (int i = startPos; i < startPos + count; i++)
        if (buffer[i] != byte)
            return false;
    return true;
}

DatalinkProtocol::DatalinkProtocol(const char *host, int port, int timeout_ms)
{
    this->conn = std::make_unique<DatalinkClientConnection>(timeout_ms);
    this->host = strdup(host);
    this->port = port;
    this->_isServer = false;
    initialize();
}
DatalinkProtocol::DatalinkProtocol(int port, int timeout_ms)
{
    this->conn = std::make_unique<DatalinkServerConnection>(timeout_ms);
    this->host = nullptr;
    this->port = port;
    this->_isServer = true;
    initialize();
}
DatalinkProtocol::~DatalinkProtocol()
{
    if (host != nullptr)
        free(host);
    closeDataConnection();
    closeListen();
}
void DatalinkProtocol::initialize()
{
    this->bufferOnUpdate = false;
    this->requestLockBufferUpdate = false;

    protocolBufferRcv = std::make_unique<char[]>(PROTOCOL_START_MESSAGE_SIZE);
    for (int i = 0; i < 5; i++)
    {
        messageStart[i] = 0x20;
        messageStart[PROTOCOL_START_MESSAGE_SIZE - i - 1] = 0x21;
    }
}
bool DatalinkProtocol::readFromSocket(char **buffer, long size)
{
    bool readRes = false;

    if (this->_isServer)
        return ((DatalinkServerConnection *)(this->conn).get())->readFromSocket(*buffer, size);
    else
        return ((DatalinkClientConnection *)(this->conn).get())->readFromSocket(*buffer, size);
}
bool DatalinkProtocol::writeToSocket(char *buffer, long size)
{
    if (this->_isServer)
        return ((DatalinkServerConnection *)(this->conn).get())->writeToSocket(buffer, size);
    else
        return ((DatalinkClientConnection *)(this->conn).get())->writeToSocket(buffer, size);
}
long DatalinkProtocol::checkMessageStart()
{
    char *buffer = protocolBufferRcv.get();

    if (!readFromSocket(&buffer, PROTOCOL_START_MESSAGE_SIZE))
    {
#ifdef DEBUG
        printf("[datalink] checkMessageRcvStart failed\n");
#endif
        return -1;
    }

    if (!checkRepeatByte(buffer, 0, 5, 32))
        return false;

    longp res;
    for (int i = 5; i < 5 + sizeof(long); i++)
        res.bval[i - 5] = buffer[i];

    if (!checkRepeatByte(buffer, 5 + sizeof(long), 5, 33))
        return false;

#ifdef DEBUG
    printf("[datalink] message start detected with %ld bytes\n", res.val);
#endif

    return res.val;
}
bool DatalinkProtocol::checkMessageEnd()
{
    char *buffer = protocolBufferRcv.get();

    if (!readFromSocket(&buffer, PROTOCOL_END_MESSAGE_SIZE))
    {
        //printBuffer(buffer, PROTOCOL_END_MESSAGE_SIZE);
        return false;
    }

    if (!checkRepeatByte(buffer, 0, 5, 34))
        return false;

    if (!checkRepeatByte(buffer, 5, 5, 35))
        return false;

#ifdef DEBUG
    printf("[datalink] message end successfully detected\n");
#endif

    return true;
}

bool DatalinkProtocol::writeMessageStart(long messageSize)
{
    longp size;
    size.val = messageSize;

    for (int i = 5; i < 5 + sizeof(long); i++)
        messageStart[i] = size.bval[i - 5];

    return writeToSocket(messageStart, PROTOCOL_START_MESSAGE_SIZE);
}
bool DatalinkProtocol::writeMessageEnd()
{
    return writeToSocket((char *)&messageEnd, PROTOCOL_END_MESSAGE_SIZE);
}

bool DatalinkProtocol::updateBuffer(long size)
{
    if (this->lastBuffer == nullptr)
    {
        this->lastBuffer = std::make_unique<DataLinkResult<char>>();
        if (this->lastBuffer == nullptr)
            return false;

        this->lastBuffer->size = 0;
    }

    if (this->lastBuffer->size != size)
    {
        // printf("updateBuffer() modifying the data buffer from %ld to %ld\n", this->lastBuffer->size, size);
        this->lastBuffer->data = std::make_unique<char[]>(size + 1);
        this->lastBuffer->size = size;
    }

    return this->lastBuffer->data != nullptr;
}

bool DatalinkProtocol::readMessage()
{        
    this->bufferOnUpdate = false;

    if (!conn->isConnected())
        return false;

    if (this->requestLockBufferUpdate)
        return false;

    this->bufferOnUpdate = true;

    long size = checkMessageStart();
    if (size == 0)
    {
        if (checkMessageEnd()) {
            conn->rstTimeout(); // its a ping message
        }

        this->bufferOnUpdate = false;
        return false;
    }
    if (size < 0)
    {
        this->bufferOnUpdate = false;
        return false;
    }

    conn->rstTimeout();

    if (lastBuffer != nullptr)
        lastBuffer->valid = false;

    if (!updateBuffer(size))
    {
        this->bufferOnUpdate = false;
        return false;
    }

    char *dataBufferPtr = this->lastBuffer->data.get();

    if (!readFromSocket(&dataBufferPtr, size))
    {
        this->bufferOnUpdate = false;
        return false;
    }

    dataBufferPtr[size] = 0x0;

    if (checkMessageEnd())
    {
        conn->rstTimeout();
        lastBuffer->valid = true;
    }

    this->bufferOnUpdate = false;
    return lastBuffer->valid;
}
bool DatalinkProtocol::sendPingMessage()
{
    if (!conn->isConnected())
        return false;

    if (!writeMessageStart(0))
        return false;

    return writeMessageEnd();
}

bool DatalinkProtocol::writeMessage(char *data, long size)
{
    if (!conn->isConnected())
        return false;

    if (!writeMessageStart(size))
        return false;

    if (!writeToSocket(data, size))
        return false;

    return writeMessageEnd();
}
bool DatalinkProtocol::writeMessage(float *data, long size)
{
    if (!conn->isConnected())
        return false;

    char *tempBuffer = new char[sizeof(float) * size];
    if (tempBuffer == nullptr)
        return false;

    floatp p;
    for (long i = 0, j = 0; i < size; i++)
    {
        p.fval = data[i];
        for (int k = 0; k < sizeof(float); k++)
        {
            tempBuffer[j++] = p.bval[k];
        }
    }

    // printBuffer(tempBuffer, sizeof(float) * size);

    bool res = writeMessage(tempBuffer, sizeof(float) * size);
    delete []tempBuffer;
    return res;
}

bool DatalinkProtocol::writeMessage(uint8_t *data, long size)
{
    if (!conn->isConnected())
        return false;

    char *tempBuffer = new char[sizeof(uint8_t) * size];
    if (tempBuffer == nullptr)
        return false;

    uint8 p;
    for (long i = 0, j = 0; i < size; i++)
    {
        p.val = data[i];
        for (int k = 0; k < sizeof(uint8_t); k++)
        {
            tempBuffer[j++] = p.bval;
        }
    }

    // printBuffer(tempBuffer, sizeof(float) * size);

    bool res = writeMessage(tempBuffer, sizeof(uint8_t) * size);
    delete tempBuffer;
    return res;
}
std::shared_ptr<DataLinkResult<char>> DatalinkProtocol::releaseBuffer()
{
    this->requestLockBufferUpdate = true;
    while (this->bufferOnUpdate)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (this->lastBuffer == nullptr)
    {
        this->requestLockBufferUpdate = false;
        return nullptr;
    }

    if (!this->lastBuffer->valid)
    {
        this->requestLockBufferUpdate = false;
        return nullptr;
    }

    DataLinkResult<char> *p = new DataLinkResult<char>();
    p->size = lastBuffer->size;
    p->valid = true;
    p->data = std::make_unique<char[]>(lastBuffer->size + 1);
    memcpy(p->data.get(), lastBuffer->data.get(), p->size);

    this->lastBuffer->valid = false;
    this->requestLockBufferUpdate = false;
    return std::shared_ptr<DataLinkResult<char>>(p);
}

bool DatalinkProtocol::hasData()
{
    return this->lastBuffer != nullptr && this->lastBuffer->valid;
}

bool DatalinkProtocol::isServer()
{
    return this->_isServer;
}

bool DatalinkProtocol::bindConnection()
{
    if (!this->_isServer)
        return false;

    return ((DatalinkServerConnection *)(this->conn).get())->bindConnection(port);
}
bool DatalinkProtocol::openConnection()
{
    if (this->_isServer)
        return false;

    DatalinkClientConnection *connection = (DatalinkClientConnection *)(this->conn).get();

    bool p = connection->openConnection(host, port);
    connection->rstTimeout();
    return p;
}
bool DatalinkProtocol::socketAcceptIncommingConnections()
{
    if (!this->_isServer)
        return false;

    DatalinkServerConnection *connection = (DatalinkServerConnection *)(this->conn).get();

    bool p = connection->socketAcceptIncommingConnections();
    if (p)
        connection->rstTimeout();
    return p;
}
bool DatalinkProtocol::checkTimeout()
{
    return this->conn->checkTimeout();
}

void DatalinkProtocol::closeDataConnection()
{
    if (this->_isServer)
        ((DatalinkServerConnection *)(this->conn).get())->closeConnection();
    else
        ((DatalinkClientConnection *)(this->conn).get())->closeConnection();
}
void DatalinkProtocol::closeListen()
{
    if (!this->_isServer)
        return;

    ((DatalinkServerConnection *)(this->conn).get())->closeListen();
}