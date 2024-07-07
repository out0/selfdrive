#ifndef __CPP_TEST_UTILS_H
#define __CPP_TEST_UTILS_H

#include <stdio.h>
#include <thread>
#include <memory>
#include <functional>
#include <chrono>
#include <iostream>

#include "../datalink_server_connection.h"
#include "../datalink_client_connection.h"
#include "../datalink_protocol.h"
#include "../datalink.h"

typedef std::unique_ptr<std::thread> thr;

static long long time_now()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::tuple<DatalinkServerConnection *, DatalinkClientConnection *> makeConnectedPair(int port, int timeout_ms = 50)
{
    auto server = new DatalinkServerConnection(timeout_ms);
    auto client = new DatalinkClientConnection(timeout_ms);

    if (!server->bindConnection(port))
        return {nullptr, nullptr};

    std::thread listenThr = std::thread([](DatalinkServerConnection *s)
                                        {
        while (!s->socketAcceptIncommingConnections()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } },
                                        server);

    while (!client->openConnection("127.0.0.1", port))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    listenThr.join();

    return {server, client};
}
static std::tuple<DatalinkProtocol *, DatalinkProtocol *> makeConnectedProtocolPair(int port, int timeout_ms = 50)
{
    DatalinkProtocol *server = new DatalinkProtocol(port, timeout_ms);
    DatalinkProtocol *client = new DatalinkProtocol("127.0.0.1", port, timeout_ms);

    if (!server->bindConnection())
        return {nullptr, nullptr};

    std::thread listenThr = std::thread([](DatalinkProtocol *s)
                                        {
        while (!s->socketAcceptIncommingConnections()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } },
                                        server);

    while (!client->openConnection())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    listenThr.join();

    return {server, client};
}

static std::tuple<Datalink *, Datalink *> makeConnectedDatalinkPair(int port, int timeout_ms = 50)
{
    Datalink *server = new Datalink(port, timeout_ms);
    Datalink *client = new Datalink("127.0.0.1", port, timeout_ms);

    if (!server->waitReady(timeout_ms * 10))
        return {nullptr, nullptr};
    if (!client->waitReady(timeout_ms * 10))
        return {nullptr, nullptr};

    return {server, client};
}

static void terminatePairProtocol(std::tuple<DatalinkProtocol *, DatalinkProtocol *> connections)
{
    DatalinkProtocol *protServer;
    DatalinkProtocol *protClient;

    std::tie(protServer, protClient) = connections;

    delete protServer;
    delete protClient;
}

static void terminatePairConnection(std::tuple<DatalinkServerConnection *, DatalinkClientConnection *> connections)
{
    DatalinkServerConnection *server = nullptr;
    DatalinkClientConnection *client = nullptr;

    std::tie(server, client) = connections;

    if (server != nullptr)
    {
        server->closeConnection();
        server->closeListen();
        ASSERT_FALSE(server->isConnected());
        delete server;
    }

    if (client != nullptr)
    {
        client->closeConnection();
        ASSERT_FALSE(client->isConnected());
        delete client;
    }
}

static void terminatePairDatalink(std::tuple<Datalink *, Datalink *> connections)
{
    Datalink *server;
    Datalink *client;

    std::tie(server, client) = connections;

    delete client;
    delete server;
}

// static thr parallelSingleExecute(std::function<void()> method)
// {
//     return thr(new std::thread(method));
// }
// static thr parallelExecuteUntilSuccessOrTimeout(std::function<bool()> method, int timeout_ms)
// {
//     return thr(new std::thread([method] (int timeout)
//                                {
//                                 long long start = time_now();
//                                    while (!method())
//                                    {
//                                        std::this_thread::sleep_for(std::chrono::milliseconds(2));

//                                        if ((int)(time_now() - start) > timeout)
//                                        {
//                                            printf("[test] timeout\n");
//                                            return;
//                                        }

//                                    } }, timeout_ms));
// }

// static void parallelWaitFinish(thr &t)
// {
//     t->join();
// }

// static bool executeUntilSuccessOrTimeout(std::function<bool()> method, int timeout_ms)
// {
//     long long start = time_now();
//     while (!method())
//     {
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         if (time_now() - start > timeout_ms)
//         {
//             printf("[test] timeout\n");
//             return false;
//         }
//     }

//     return true;
// }

static void wait_ms(int ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

#endif
