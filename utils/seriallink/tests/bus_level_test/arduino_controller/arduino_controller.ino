#include <seriallink.h>
#include <string.h>
#include "serial_bus_arduino.h"

SerialBusArduino bus(1000);
char buff[512];

void setup() {
  // put your setup code here, to run once:
  bus.initialize();
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  
  if (bus.dataAvail() > 0) {
     unsigned int s = bus.readBytesTo(buff, 0, 512);
     bus.writeBytes(buff, s);
  }
}
