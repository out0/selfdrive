#include <seriallink.h>
#include <string.h>
#include "serial_bus_arduino.h"
#include "actuators/double_ackman.h"
#include "actuators/engine_power.h"
#include "crawler_protocol.h"

//#define ENABLE_DEBUG_ECHO 1

SerialBusArduino bus(1000);
SerialLink link(&bus, 512);
DoubleAckman steering;
EnginePower engine;

void executeCommand(unsigned char cmd);

void setup()
{
  // put your setup code here, to run once:
  bus.initialize();
  steering.initialize();
  engine.initialize();
}

void loop()
{
  link.loop();

  if (link.hasData())
  {
    unsigned char cmd = link.readByte();
    executeCommand(cmd);
  }
}

void executeCommand(unsigned char cmd)
{
  float val = 0.0;

  switch (cmd)
  {
  case CMD_RST_ACTUATORS:
    engine.stop();
    steering.setSteeringAngle(0);
    break;
  case CMD_SET_POWER:
    val = link.readFloat();
    engine.setPower(val);
    break;
  case CMD_SET_STEERING_ANGLE:
    val = link.readFloat();
    steering.setSteeringAngle(val);
    break;
  case CMD_STOP:
    engine.stop();
    break;
  default:
    break;
  }
  link.endRead();

#ifdef ENABLE_DEBUG_ECHO
  link.writeByte(DATA_CMD_ECHO);
  link.writeByte(cmd);
  link.write(val);
  link.send();
#endif

  bus.waitBus();
  link.sendAck();
}