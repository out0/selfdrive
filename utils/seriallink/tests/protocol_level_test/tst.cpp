#include <stdlib.h>
#include <serial_bus_linux.h>
#include <serial_protocol.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <stdio.h>


char msg[] = "HELLO CP/500!!!";

int main(int argc, char **argv)
{
    SerialBusLinux bus("/dev/ttyACM0", 200);
    SerialLinkProtocol protocol(&bus, 512);

    bus.initialize();
    printf("sent: %s\n", msg);
    protocol.writeString(msg, strlen(msg));
    protocol.sendData();
        
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    protocol.receiveData();
    unsigned int len = protocol.rcvSize();
    char * p = protocol.readString(len);
    printf("received: %s\n", p);
    protocol.flushRecvBuffer();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    protocol.receiveData();
    
    if (protocol.checkMessageAck()) {
        printf("Message acked\n");
    } else {
        printf("Message acked failed\n");
    }
}