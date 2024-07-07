#include "../datalink.h"
#include <unistd.h>

#define SIZE 2097100
int main()
{
    
    auto link = Datalink(21001);
    int data_count = 0;

    char *huge_data = new char[SIZE];
    for (int i = 0; i < SIZE; i++)
        huge_data[i] = i % 100;

    while (true)
    {
        if (!link.isReady()) {
            printf ("link not ready\n");
            sleep(1);
            continue;
        }
        printf("sending data %d\n", data_count);
        link.sendData(huge_data, SIZE);

        data_count++;
        //sleep(1);
    }

    delete []huge_data;
    
}