#include <string>
#include <memory>
#include <cstring>
#include "../datalink.h"

// #define DEBUG_BIND 1

extern "C"
{
        void *init_server(int port, int timeout_ms)
        {
                auto link = new Datalink(port, timeout_ms);
#ifdef DEBUG_BIND
                printf("creating a Datalink server %p with port: %d\n", link, port);
#endif
                return link;
        }

        void *init_client(char *host, int port, int timeout_ms)
        {
                auto link = new Datalink(host, port, timeout_ms);
#ifdef DEBUG_BIND
                printf("creating a Datalink client %p to host %s and port: %d\n", link, host, port);
#endif
                return link;
        }

        void destroy_link(void *link)
        {
#ifdef DEBUG_BIND
                printf("destroying the Datalink %p\n", link);
#endif
                delete ((Datalink *)link);
        }

        bool write_str_link(void *link, char *data, long size)
        {
#ifdef DEBUG_BIND
                printf("writing string to the Datalink %p size: %ld\n", link, size);
#endif
                return ((Datalink *)link)->sendData(data, size);
        }

        bool write_np_float_link(void *link, float *input_array, size_t size)
        {
                return ((Datalink *)link)->sendData(input_array, size);
        }
        bool write_np_uint8_link(void *link, uint8_t *input_array, size_t size)
        {
                return ((Datalink *)link)->sendData(input_array, size);
        }

        bool has_data(void *link)
        {
                return ((Datalink *)link)->hasData();
        }

        bool wait_ready(void *link, int timeout_ms)
        {
                return ((Datalink *)link)->waitReady(timeout_ms);
        }
        bool is_connected(void *link)
        {
                if (link == nullptr)
                {
                        printf("is_connected() called with a null pointer\n\n");
                        return false;
                }
                return ((Datalink *)link)->isReady();
        }

        bool is_listening(void *link)
        {
                return ((Datalink *)link)->isReady();
        }

        char *read_str_link(void *link, long *size, int timeout_ms)
        {
                auto ptr = ((Datalink *)link);
#ifdef DEBUG_BIND
                printf("reading string from the Datalink %p\n", link);
#endif
                auto res = ptr->receiveData(timeout_ms);
                if (res == nullptr)
                {
                        *size = 0;
                        return nullptr;
                }
                *size = res->size;
                return res->data.release();
        }

        float *read_np_float_link(void *link, long *size, int timeout_ms)
        {
#ifdef DEBUG_BIND
                printf("reading float array from the Datalink %p\n", link);
#endif
                auto ptr = ((Datalink *)link);
                auto res = ptr->receiveDataF(timeout_ms);
                if (res == nullptr)
                {
                        *size = 0;
                        return nullptr;
                }

                *size = res->size;
                return res->data.release();
        }

        void free_memory(void *address)
        {
#ifdef DEBUG_BIND
                printf("freeing memory at %p\n", address);
#endif
                free(address);
        }

        void terminate(void *link) {
                auto ptr = ((Datalink *)link);
                ptr->destroy();
        }
}