#include <gtest/gtest.h>
#include <stdlib.h>
#include "../datalink_client_connection.h"
#include "../datalink_server_connection.h"
#include <tuple>
#include <iostream>
#include "cpp_test_utils.h"

TEST(DataLinkConnectionTest, TestWrongConnection)
{
    auto client = new DatalinkClientConnection(500);
    ASSERT_FALSE(client->openConnection("127.0.0.1", 20000));
    delete client;
}

TEST(DataLinkConnectionTest, TestSimpleConnection)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000);
    if (server == nullptr || client == nullptr)
        FAIL();

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestTimeout)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000, 100);

    if (server == nullptr || client == nullptr)
        FAIL();

    ASSERT_FALSE(server->checkTimeout());
    // ensure timeout
    wait_ms(120);

    ASSERT_TRUE(server->checkTimeout());

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestTimeoutReset)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000, 100);
    if (server == nullptr || client == nullptr)
        FAIL();

    ASSERT_FALSE(server->checkTimeout());

    for (int i = 0; i < 3; i++)
    {
        wait_ms(50);
        server->rstTimeout();
    }

    ASSERT_FALSE(server->checkTimeout());

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestDataSendServer2Client)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000);
    if (server == nullptr || client == nullptr)
        FAIL();

    char buffer[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};

    server->writeToSocket(buffer, sizeof(char) * 5);

    char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    wait_ms(10);
    client->readFromSocket(outBuffer, sizeof(char) * 5);

    ASSERT_STREQ(buffer, outBuffer);

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestDataSendClient2Server)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000);
    if (server == nullptr || client == nullptr)
        FAIL();

    char buffer[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};

    ASSERT_TRUE(client->writeToSocket(buffer, sizeof(char) * 5));

    char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    wait_ms(10);
    ASSERT_TRUE(server->readFromSocket(outBuffer, sizeof(char) * 5));

    ASSERT_STREQ(buffer, outBuffer);

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestDataSendBiDirectional)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;
    std::tie(server, client) = makeConnectedPair(2000);
    if (server == nullptr || client == nullptr)
        FAIL();

    char buffer1[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};
    char buffer2[] = {0x16, 0x17, 0x18, 0x19, 0x20, 0x0};

    ASSERT_TRUE(client->isConnected());
    ASSERT_TRUE(server->isConnected());

    ASSERT_TRUE(client->writeToSocket(buffer1, sizeof(char) * 5));

    ASSERT_TRUE(server->writeToSocket(buffer2, sizeof(char) * 5));

    char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    wait_ms(10);

    server->readFromSocket(outBuffer, sizeof(char) * 5);
    ASSERT_STREQ(buffer1, outBuffer);

    client->readFromSocket(outBuffer, sizeof(char) * 5);
    ASSERT_STREQ(buffer2, outBuffer);

    terminatePairConnection({server, client});
}

TEST(DataLinkConnectionTest, TestManyParallelConnections)
{
    int parallelCount = 1000;
    std::tuple<DatalinkServerConnection *, DatalinkClientConnection *> connections[parallelCount];

    for (int i = 0; i < parallelCount; i++)
    {
        connections[i] = makeConnectedPair(20000 + i, 2000);
    }

    char buffer1[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};
    char buffer2[] = {0x16, 0x17, 0x18, 0x19, 0x20, 0x0};

    for (int i = 0; i < parallelCount; i++)
    {
        DatalinkServerConnection *server = nullptr;
        DatalinkClientConnection *client = nullptr;
        std::tie(server, client) = connections[i];
        if (server == nullptr || client == nullptr)
            FAIL();

        client->writeToSocket(buffer1, 5);
        server->writeToSocket(buffer2, 5);

        char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

        wait_ms(1);

        server->readFromSocket(outBuffer, sizeof(char) * 5);
        ASSERT_STREQ(buffer1, outBuffer);

        client->readFromSocket(outBuffer, sizeof(char) * 5);
        ASSERT_STREQ(buffer2, outBuffer);
    }

    for (int i = 0; i < parallelCount; i++)
    {
        terminatePairConnection(connections[i]);
    }
}