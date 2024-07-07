#include "databridge.h"

DataBrigde::DataBrigde(const char *inHost, int inPort, const char *outHost, int outPort, int timeout_ms)
    : Datalink(inHost, inPort)
{
    this->outLink = new Datalink(outHost, outPort, timeout_ms);
}

DataBrigde::DataBrigde(int inPort, const char *outHost, int outPort, int timeout_ms)
    : Datalink(inPort)
{
    this->outLink = new Datalink(outHost, outPort, timeout_ms);
}

DataBrigde::DataBrigde(int inPort, int outPort, int timeout_ms)
    : Datalink(inPort)
{
    this->outLink = new Datalink(outPort, timeout_ms);
}

DataBrigde::DataBrigde(const char *inHost, int inPort, int outPort, int timeout_ms)
    : Datalink(inHost, inPort)
{
    this->outLink = new Datalink(outPort, timeout_ms);
}

void DataBrigde::onDataReceived() {
    if (this->outLink->isReady())  {
        auto res = receiveData();
        this->outLink->sendData(res->data.get(), res->size);
    }
}

DataBrigde::~DataBrigde() {
    delete this->outLink;
}
