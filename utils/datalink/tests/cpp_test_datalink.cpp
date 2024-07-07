#include <gtest/gtest.h>
#include <stdlib.h>
#include "../datalink.h"
#include "cpp_test_utils.h"

TEST(DataLinkTest, TestCreateDestroy)
{
    int count = 2;
    Datalink **p = new Datalink *[2 * count];

    printf("creating %d connection pairs\n", count);
    for (int i = 0; i < count; i++)
    {
        p[i] = new Datalink(20000 + i);
        p[count + i] = new Datalink("127.0.0.1", 20000 + i);
    }

    for (int i = 0; i < count; i++)
    {
        EXPECT_TRUE(p[i]->waitReady(2000));
        EXPECT_TRUE(p[count + i]->waitReady(2000));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    printf("destroying %d connection pairs\n", count);
    for (int i = 0; i < count; i++)
    {
        p[i]->destroy();
        p[count + i]->destroy();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    for (int i = 0; i < count; i++)
    {
        delete p[i];
        delete p[count + i];
    }

    delete[] p;
}

TEST(DataLinkTest, TestSendMessageServer2Client)
{
    char msg[] = {0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x00};

    Datalink *srv;
    Datalink *cli;
    auto pair = makeConnectedDatalinkPair(20000);
    std::tie(srv, cli) = pair;

    EXPECT_FALSE(srv == nullptr);

    if (srv == nullptr || cli == nullptr)
    {
        if (srv != nullptr)
            delete srv;
        if (cli != nullptr)
            delete cli;
        return;
    }

    EXPECT_TRUE(srv->sendData(msg, 10));

    auto result = cli->receiveData(2000);

    EXPECT_FALSE(result == nullptr);

    EXPECT_TRUE(result->valid);
    EXPECT_EQ(result->size, 10);
    EXPECT_STREQ(result->data.get(), msg);

    terminatePairDatalink(pair);
}

TEST(DataLinkTest, TestSendMessageServer2ClientFloat)
{
    int size = 1000000;
    float *msg = new float[size];

    for (int i = 0; i < size; i++)
        msg[i] = i * 1.001;

    Datalink *srv;
    Datalink *cli;
    auto pair = makeConnectedDatalinkPair(20000);
    std::tie(srv, cli) = pair;
    EXPECT_FALSE(srv == nullptr);

    if (srv == nullptr || cli == nullptr)
    {
        if (srv != nullptr)
            delete srv;
        if (cli != nullptr)
            delete cli;
        return;
    }

    EXPECT_TRUE(srv->sendData(msg, size));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto result = cli->receiveDataF(2000);

    EXPECT_FALSE(result == nullptr);

    if (result != nullptr)
    {

        EXPECT_TRUE(result->valid);
        EXPECT_EQ(result->size, size);

        for (int i = 0; i < size; i++)
            if (result->data[i] != (float)(i * 1.001))
            {
                printf("data is corrupted at %d: expected %f obtained %f\n", i, i * 1.001, result->data[i]);
                FAIL();
            }
    }

    terminatePairDatalink(pair);
    delete[] msg;
}

TEST(DataLinkTest, TestManyParallelConnections)
{
    int parallelCount = 2;
    std::tuple<Datalink *, Datalink *> connections[parallelCount];

    for (int i = 0; i < parallelCount; i++)
    {
        connections[i] = makeConnectedDatalinkPair(20000 + i, 500);

        if (std::get<0>(connections[0]) == nullptr)
        {
            printf("error establishing a pair connection for #%d\n", i);
            FAIL();
        }
    }

    printf("all connections were ok\n");

    char buffer1[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};
    char buffer2[] = {0x16, 0x17, 0x18, 0x19, 0x20, 0x0};

    for (int i = 0; i < parallelCount; i++)
    {
        Datalink *server = nullptr;
        Datalink *client = nullptr;
        std::tie(server, client) = connections[i];

        printf("client send\n");
        client->sendData(buffer1, 5);
        printf("server send\n");
        server->sendData(buffer2, 5);

        char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

        wait_ms(1);

        printf("client receive\n");
        auto buffer = client->receiveData(100);
        EXPECT_TRUE(buffer != nullptr);

        if (buffer != nullptr)
        {
            EXPECT_TRUE(buffer->valid);
            EXPECT_STREQ(buffer->data.get(), buffer2);
            EXPECT_EQ(buffer->size, 5);
        }
        printf("server receive\n");
        buffer = server->receiveData(100);
        EXPECT_TRUE(buffer != nullptr);

        if (buffer != nullptr)
        {
            EXPECT_TRUE(buffer->valid);
            EXPECT_STREQ(buffer->data.get(), buffer1);
            EXPECT_EQ(buffer->size, 5);
        }
    }

    for (int i = 0; i < parallelCount; i++)
    {
        std::get<0>(connections[i])->destroy();
        std::get<1>(connections[i])->destroy();
    }

    for (int i = 0; i < parallelCount; i++)
    {
        printf("terminating pair #d%d\n", i);
        terminatePairDatalink(connections[i]);
    }
}
