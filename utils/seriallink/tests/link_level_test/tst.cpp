#include <stdlib.h>
#include <serial_bus_linux.h>
#include <seriallink.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <stdio.h>


char msg[] = "HELLO CP/500!!!";

int main(int argc, char **argv)
{
    SerialBusLinux bus("/dev/ttyACM0", 1000);
    SerialLink link(&bus, 512);

    bus.initialize();
    printf("sent: %s\n", msg);
    link.write(msg, strlen(msg));
    link.send();
        
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    link.loop();

    unsigned int len = link.dataSize();
    char * p = link.readString(len);
    printf("received: %s\n", p);
    link.endRead();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    if (link.waitAck(true)) {
        printf("Message acked\n");
    } else {
        printf("Message acked failed\n");
    }
}