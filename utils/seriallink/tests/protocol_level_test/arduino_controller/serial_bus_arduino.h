#include <serial_bus.h>
#include <Arduino.h>
#include <stdint.h>

#define SERIAL_BOUND_RATE 38400
#define SERIAL_RCV_WAIT_DELAY_ms 2
#define RCV_BUFFER_SIZE 64
#define SND_BUFFER_SIZE 64

#define min(x, y) x < y ? x : y

class SerialBusArduino : public SerialBus
{
    unsigned long timeout_st;
    int timeout_ms;

public:
    SerialBusArduino(int timeout_ms) : timeout_ms(timeout_ms),
                                       timeout_st(millis())
    {
    }

    void initialize()
    {
        Serial.begin(SERIAL_BOUND_RATE);
    }

    int dataAvail()
    {
        return Serial.available();
    }

    unsigned char readByte()
    {
        return Serial.read();
    }

    unsigned int readBytesTo(char *buffer, int start, int maxSize)
    {
        unsigned int s = min(maxSize, dataAvail());
        // for (unsigned int i = 0; i < s; i++)
        // {
        //     *buffer[start + i] = Serial.read();
        // }
        return Serial.readBytes(buffer, s);
    }

    void writeByte(unsigned char val)
    {
        Serial.write(val);
    }

    void writeBytes(char *buffer, int size)
    {
      Serial.write(buffer, size);
    }

    void flush()
    {
        Serial.flush();
    }

    bool isReady()
    {
        return (Serial);
    }

    void waitBus()
    {
        delay(SERIAL_RCV_WAIT_DELAY_ms);
    }

    void rstTimeout() {
        timeout_st = millis();
    }

    bool checkTimeout() {
        return (millis() - timeout_st) > timeout_ms;
    }
};