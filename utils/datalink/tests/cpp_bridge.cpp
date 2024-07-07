#include "../databridge.h"
#include <unistd.h>

#define SIZE 2097100
int main()
{
    auto bridge = DataBrigde("127.0.0.1", 21001, 25000, 0.5);
    
    while(true) {
        sleep(1);
    }
}