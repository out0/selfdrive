#include <gtest/gtest.h>
#include <stdlib.h>
#include "../datalink_client_connection.h"
#include "../datalink_server_connection.h"
#include "../datalink_protocol.h"
#include "cpp_test_utils.h"
#include <tuple>

TEST(DataLinkProtocolTest, TestSendMessageServer2Client)
{
    char msg[] = {0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x00};

    DatalinkProtocol *srv;
    DatalinkProtocol *cli;
    auto pair = makeConnectedProtocolPair(20000);
    std::tie(srv, cli) = pair;

    ASSERT_TRUE(srv->writeMessage(msg, 10));
    wait_ms(5);

    ASSERT_TRUE(cli->readMessage());
    auto result = cli->releaseBuffer();
    ASSERT_TRUE(result->valid);
    ASSERT_EQ(result->size, 10);
    ASSERT_STREQ(result->data.get(), msg);

    terminatePairProtocol(pair);
}

TEST(DataLinkProtocolTest, TestSendMessageClient2Server)
{
    char msg[] = {0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x00};
    DatalinkProtocol *srv;
    DatalinkProtocol *cli;

    auto pair = makeConnectedProtocolPair(20000);
    std::tie(srv, cli) = pair;

    ASSERT_TRUE(cli->writeMessage(msg, 10));

    wait_ms(5);
    ASSERT_TRUE(srv->readMessage());
    auto result = srv->releaseBuffer();
    ASSERT_TRUE(result->valid);
    ASSERT_EQ(result->size, 10);
    ASSERT_STREQ(result->data.get(), msg);

    terminatePairProtocol(pair);
}
TEST(DataLinkProtocolTest, TestSendMessageBidirectional)
{
    char msg[] = {0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x00};
    DatalinkProtocol *srv;
    DatalinkProtocol *cli;
    auto pair = makeConnectedProtocolPair(20000);
    std::tie(srv, cli) = pair;

    ASSERT_TRUE(cli->writeMessage(msg, 10));
    ASSERT_TRUE(srv->writeMessage(msg, 10));

    ASSERT_TRUE(cli->readMessage());

    ASSERT_TRUE(srv->readMessage());

    auto result = srv->releaseBuffer();
    ASSERT_TRUE(result->valid);
    ASSERT_EQ(result->size, 10);
    ASSERT_STREQ(result->data.get(), msg);

    auto result2 = cli->releaseBuffer();
    ASSERT_TRUE(result2->valid);
    ASSERT_EQ(result2->size, 10);
    ASSERT_STREQ(result2->data.get(), msg);

    terminatePairProtocol(pair);
}

TEST(DataLinkProtocolTest, TestSendMessageAfterBufferRelease)
{
    char msg[] = {0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x11, 0x12, 0x00};
    DatalinkProtocol *srv;
    DatalinkProtocol *cli;
    auto pair = makeConnectedProtocolPair(20000);
    std::tie(srv, cli) = pair;

    for (int i = 0; i < 5; i++)
    {
        ASSERT_TRUE(cli->writeMessage(msg, 10));
        wait_ms(5);
        ASSERT_TRUE(srv->readMessage());
        auto result = srv->releaseBuffer();
        ASSERT_TRUE(result->valid);
        ASSERT_EQ(result->size, 10);
        ASSERT_STREQ(result->data.get(), msg);
    }

    terminatePairProtocol(pair);
}

TEST(DataLinkProtocolTest, TestManyParallelConnections)
{
    int parallelCount = 1000;
    std::tuple<DatalinkProtocol *, DatalinkProtocol *> connections[parallelCount];

    for (int i = 0; i < parallelCount; i++)
    {
        connections[i] = makeConnectedProtocolPair(20000 + i, 20000);

        if (std::get<0>(connections[0]) == nullptr)
        {
            printf("error establishing a pair connection for #%d\n", i);
            FAIL();
        }
    }

    char buffer1[] = {0x11, 0x12, 0x13, 0x14, 0x15, 0x0};
    char buffer2[] = {0x16, 0x17, 0x18, 0x19, 0x20, 0x0};

    for (int i = 0; i < parallelCount; i++)
    {
        DatalinkProtocol *server = nullptr;
        DatalinkProtocol *client = nullptr;
        std::tie(server, client) = connections[i];
        client->writeMessage(buffer1, 5);
        server->writeMessage(buffer2, 5);

        char outBuffer[]{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

        wait_ms(1);
        ASSERT_TRUE(server->readMessage());
        ASSERT_TRUE(client->readMessage());

        auto buffer = server->releaseBuffer();
        ASSERT_TRUE(buffer->valid);
        ASSERT_STREQ(buffer->data.get(), buffer1);
        ASSERT_EQ(buffer->size, 5);

        buffer = client->releaseBuffer();
        ASSERT_TRUE(buffer->valid);
        ASSERT_STREQ(buffer->data.get(), buffer2);
        ASSERT_EQ(buffer->size, 5);
    }

    for (int i = 0; i < parallelCount; i++)
    {
        terminatePairProtocol(connections[i]);
    }
}

TEST(DataLinkProtocolTest, TestConnectionKeeptWithPing)
{
    DatalinkProtocol *srv;
    DatalinkProtocol *cli;
    auto pair = makeConnectedProtocolPair(20000);
    std::tie(srv, cli) = pair;

    ASSERT_FALSE(srv->checkTimeout());
    ASSERT_FALSE(cli->checkTimeout());

    for (int i = 0; i < 20; i++)
    {
        srv->sendPingMessage();
        cli->sendPingMessage();
        wait_ms(10);
    }

    ASSERT_FALSE(srv->checkTimeout());
    ASSERT_FALSE(cli->checkTimeout());

    wait_ms(200);

    ASSERT_TRUE(srv->checkTimeout());
    ASSERT_TRUE(cli->checkTimeout());

    terminatePairProtocol(pair);
}

