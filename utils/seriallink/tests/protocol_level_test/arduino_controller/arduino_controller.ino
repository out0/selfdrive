#include <seriallink.h>
#include <string.h>
#include "serial_bus_arduino.h"

SerialBusArduino bus(1000);
SerialLink link(&bus, 512);

void setup() {
  // put your setup code here, to run once:
  bus.initialize();
}

void loop() {
  link.loop();

  if (link.hasData()) {
    unsigned int len = link.dataSize();
    char* s = link.readString(len);
    link.endRead();
    link.write(s, len);
    link.send();
    link.sendAck();
  }
}
