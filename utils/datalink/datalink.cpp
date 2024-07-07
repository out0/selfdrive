#include "datalink.h"
#include <cmath>

Datalink::Datalink(const char *host, int port, int timeout_ms)
{
    this->state = CLOSED;
    this->linkProtocol = std::make_unique<DatalinkProtocol>(host, port, timeout_ms);
    this->asyncExecutor = new std::thread(&Datalink::asyncExec, this);
    this->asyncPingKeepAlive = new std::thread(&Datalink::pingKeepAliveExec, this);

    if (timeout_ms <= 1)
        this->halfTimeout_ms = 1000;
    else
        this->halfTimeout_ms = floor(timeout_ms / 2);
}
Datalink::Datalink(int port, int timeout_ms)
{
    this->state = CLOSED;
    this->linkProtocol = std::make_unique<DatalinkProtocol>(port, timeout_ms);
    this->asyncExecutor = new std::thread(&Datalink::asyncExec, this);
    this->asyncPingKeepAlive = new std::thread(&Datalink::pingKeepAliveExec, this);

    if (timeout_ms <= 1)
        this->halfTimeout_ms = 1000;
    else
        this->halfTimeout_ms = floor(timeout_ms / 2);
}
Datalink::~Datalink()
{
    this->run = false;
    this->asyncExecutor->join();
    this->asyncPingKeepAlive->join();
    this->linkProtocol = nullptr;
    delete this->asyncExecutor;
    delete this->asyncPingKeepAlive;
}

void Datalink::asyncExec()
{
    this->run = true;
    this->state = CLOSED;
    long long dt1 = DatalinkConnection::timeNow();

    while (this->run)
    {
        dt1 = DatalinkConnection::timeNow();
        switch (state)
        {
        case CLOSED:
            if (this->linkProtocol->isServer())
                this->state = BIND_PORT;
            else
                this->state = CONNECT;
            break;
        case BIND_PORT:
            this->state = bindPortExec();
            break;
        case LISTEN:
            this->state = listenExec();
            break;
        case CONNECT:
            this->state = connectExec();
            break;
        case RECEIVE_DATA:
            this->state = receiveDataExec();
            break;
        case TERMINATE_SESSION:
            this->state = terminateSessionExec();
            break;
        default:
            break;
        }
    }
}

void Datalink::pingKeepAliveExec()
{
    while (this->run)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(halfTimeout_ms));
        if (this->state == RECEIVE_DATA)
            sendPingData();
    }
}

void Datalink::sendPingData()
{
    if (!run)
        return;

    sendMtx.lock();
    this->linkProtocol->sendPingMessage();
    sendMtx.unlock();
}

int Datalink::bindPortExec()
{
    if (this->linkProtocol->bindConnection())
    {
        return LISTEN;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return CLOSED;
}

int Datalink::listenExec()
{
    if (this->linkProtocol->socketAcceptIncommingConnections())
    {
        return RECEIVE_DATA;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return LISTEN;
}

int Datalink::connectExec()
{
    if (this->linkProtocol->openConnection())
        return RECEIVE_DATA;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return CONNECT;
}

int Datalink::receiveDataExec()
{
    if (this->linkProtocol->checkTimeout())
        return TERMINATE_SESSION;

    if (!run)
        return CLOSED;

    if (this->linkProtocol->readMessage())
        onDataReceived();

    return RECEIVE_DATA;
}
int Datalink::terminateSessionExec()
{
    this->linkProtocol->closeDataConnection();
    if (this->linkProtocol->isServer())
        return LISTEN;
    else
        return CONNECT;
}

void Datalink::destroy()
{
    this->run = false;
}

bool Datalink::isReady()
{
    return this->state == RECEIVE_DATA;
}

bool Datalink::waitReady(int timeout_ms)
{
    long long start = DatalinkConnection::timeNow();
    while (!isReady())
    {
        if ((int)(DatalinkConnection::timeNow() - start) > timeout_ms)
            return false;
    }
    return true;
}

bool Datalink::hasData()
{
    return run && this->linkProtocol->hasData();
}

bool Datalink::sendData(char *data, long dataSize)
{
    if (!run)
        return false;

    sendMtx.lock();
    bool res = this->linkProtocol->writeMessage(data, dataSize);
    sendMtx.unlock();
    return res;
}
bool Datalink::sendData(float *data, long dataSize)
{
    if (!run)
        return false;

    sendMtx.lock();
    bool res = this->linkProtocol->writeMessage(data, dataSize);
    sendMtx.unlock();
    return res;
}
bool Datalink::sendData(uint8_t *data, long dataSize)
{
    if (!run)
        return false;

    sendMtx.lock();
    bool res = this->linkProtocol->writeMessage(data, dataSize);
    sendMtx.unlock();
    return res;
}
std::shared_ptr<DataLinkResult<char>> Datalink::receiveData(int timeout_ms)
{
    if (!run)
        return nullptr;

    if (timeout_ms > 0)
    {
        long long start = DatalinkConnection::timeNow();
        while (!hasData())
        {
            if ((int)(DatalinkConnection::timeNow() - start) > timeout_ms)
                return nullptr;
        }
    }

    return this->linkProtocol->releaseBuffer();
}

std::shared_ptr<DataLinkResult<float>> Datalink::receiveDataF(int timeout_ms)
{
    std::shared_ptr<DataLinkResult<char>> rawData = receiveData(timeout_ms);
    if (rawData == nullptr)
        return nullptr;

    if (rawData->size % sizeof(float) != 0)
    {
        printf("[datalink] data mismatch when receiving float: %ld is not a multiple of float size %ld\n", rawData->size, sizeof(float));
        return nullptr;
    }

    char *buffer = rawData->data.get();
    auto res = std::make_shared<DataLinkResult<float>>();
    res->size = 0;
    res->valid = true;
    res->data = std::make_unique<float[]>(rawData->size / sizeof(float));

    floatp p;
    res->size = 0;
    for (long i = 0; i < rawData->size;)
    {
        for (int k = 0; k < sizeof(float); k++, i++)
        {
            p.bval[k] = buffer[i];
        }
        res->data[res->size] = p.fval;
        res->size++;
    }

    return res;
}
