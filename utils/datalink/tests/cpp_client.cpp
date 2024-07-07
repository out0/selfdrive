#include "../datalink.h"
#include <cstring>
#include <unistd.h>

#define SIZE 2097100
int main()
{
    
    auto link = new Datalink("127.0.0.1", 21001);
    int data_count = 0;

    link->waitReady(-1);

    while (true)
    {
        auto res = link->receiveData();
        if (res == nullptr)
            continue;

        printf("receiving data %ld data_count: %d\n", res->size, data_count);
        if (res->size != 2097100)
        {
            printf("size mismatch\n");
            exit(1);
        }

        for (int i = 0; i < SIZE; i++)
        {
            if (res->data[i] != i % 100)
            {
                printf("data is corrupt %d != %d\n", i, res->data[i]);
                exit(1);
            }
        }
        data_count++;
    }

    delete link;
    
}