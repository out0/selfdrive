#include <stdlib.h>
#include <serial_bus_linux.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <stdio.h>


char buff[512];
char msg[] = "HELLO CP/500!!!\n";

int main(int argc, char **argv)
{
    SerialBusLinux bus("/dev/ttyACM0", 200);

    bus.initialize();

    bus.writeBytes(msg, strlen(msg));
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    unsigned int p = bus.readBytesTo(buff, 0, 512);
    buff[p] = 0;
    printf("received: %s\n", buff);

    bus.rstTimeout();
    if (bus.checkTimeout()) {
        printf ("ERROR: timeout where it shouldn't be\n");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (!bus.checkTimeout()) {
        printf ("ERROR: failed to timeout\n");
    }

}