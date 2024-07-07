#ifndef __DATA_BRIGDGE_H
#define __DATA_BRIGDGE_H

#include "datalink.h"

class DataBrigde : public Datalink
{
    Datalink *outLink;
    bool run;

public:
    DataBrigde(const char *inHost, int inPort, const char *outHost, int outPort, int timeout_ms);
    DataBrigde(int inPort, const char *outHost, int outPort, int timeout_ms);
    DataBrigde(const char *inHost, int inPort, int outPort, int timeout_ms);
    DataBrigde(int inPort, int outPort, int timeout_ms);
    ~DataBrigde();
    void onDataReceived() override;
    
};

#endif