#include <gtest/gtest.h>
#include <stdlib.h>
#include "../datalink.h"
#include "cpp_test_utils.h"

TEST(DataLinkParallelConnectionsTest, TestManyParallelConnections)
{
    int connections_count = 100;
    int data_size = 102400;
    Datalink **linkServers = new Datalink *[connections_count];
    Datalink **linkClients = new Datalink *[connections_count];

    for (int i = 0; i < connections_count; i++)
    {
        linkServers[i] = new Datalink(19000 + i, 2);
    }
    for (int i = 0; i < connections_count; i++)
    {
        linkClients[i] = new Datalink("127.0.0.1", 19000 + i, 2);
    }

    for (int i = 0; i < connections_count; i++)
        if (!executeUntilSuccessOrTimeout([&]()
                                          { return linkServers[i]->isReady(); },
                                          3000))
            FAIL();

    float *floatData = new float[data_size];
    for (int i = 0; i < data_size; i++)
    {
        floatData[i] = 1.0001 * i;
    }

    uint8_t *uintData = new uint8_t[data_size];
    for (int i = 0; i < data_size; i++)
    {
        uintData[i] = 21;
    }

    for (int i = 0; i < connections_count; i++)
    {
        linkServers[i]->sendData(floatData, data_size);
    }

    for (int i = 0; i < connections_count; i++)
    {
        auto res = linkClients[i]->receiveDataF(5);

        ASSERT_EQ(res->size, data_size);

        for (int j = 0; j < data_size; j++)
        {
            if (res->data[j] != floatData[j])
            {
                printf("data was corrupt for pair connection #%d\n", i);
                FAIL();
            }
        }
    }

    for (int i = 0; i < connections_count; i++)
    {
        linkClients[i]->sendData(uintData, data_size);
    }

    for (int i = 0; i < connections_count; i++)
    {
        auto res = linkServers[i]->receiveDataF(5);

        ASSERT_EQ(res->size, data_size);

        for (int j = 0; j < data_size; j++)
        {
            if (res->data[j] != floatData[j])
            {
                printf("data was corrupt for pair connection #%d\n", i);
                FAIL();
            }
        }
    }

    for (int i = 0; i < connections_count; i++)
    {
        delete linkClients[i];
        delete linkServers[i];
    }

    delete linkServers;
    delete linkClients;
    delete floatData;
    delete uintData;
}
