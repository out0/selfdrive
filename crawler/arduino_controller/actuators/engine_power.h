#ifndef H_ENGINE
#define H_ENGINE

#include <stdint.h>
#include <Arduino.h>

#define PHYSICAL_CENTER_ANGLE_FRONT 90
#define PHYSICAL_CENTER_ANGLE_BACK 76
#define MAX_STEERING 40

#define PWM_DIRECT_FRONT_PIN 6
#define PWM_DIRECT_BACK_PIN 4
#define PWM_REVERSE_FRONT_PIN 7
#define PWM_REVERSE_BACK_PIN 5
#define ENABLE_DIRECT_FRONT_PIN 30
#define ENABLE_DIRECT_BACK_PIN 36
#define ENABLE_REVERSE_FRONT_PIN 34
#define ENABLE_REVERSE_BACK_PIN 32

#define MAX_POWER 1.0

class EnginePower
{
private:
    float currentPower;

public:
    EnginePower() : currentPower(0)
    {
    }

    void initialize()
    {
        pinMode(PWM_DIRECT_FRONT_PIN, OUTPUT);
        pinMode(PWM_DIRECT_BACK_PIN, OUTPUT);
        pinMode(PWM_REVERSE_FRONT_PIN, OUTPUT);
        pinMode(PWM_REVERSE_BACK_PIN, OUTPUT);
        pinMode(ENABLE_DIRECT_FRONT_PIN, OUTPUT);
        pinMode(ENABLE_DIRECT_BACK_PIN, OUTPUT);
        pinMode(ENABLE_REVERSE_FRONT_PIN, OUTPUT);
        pinMode(ENABLE_REVERSE_BACK_PIN, OUTPUT);

        digitalWrite(ENABLE_DIRECT_FRONT_PIN, HIGH);
        digitalWrite(ENABLE_DIRECT_BACK_PIN, HIGH);
        digitalWrite(ENABLE_REVERSE_FRONT_PIN, HIGH);
        digitalWrite(ENABLE_REVERSE_BACK_PIN, HIGH);
    }
    void stop()
    {
        this->currentPower = 0;
        analogWrite(PWM_DIRECT_FRONT_PIN, 0);
        analogWrite(PWM_DIRECT_BACK_PIN, 0);
        analogWrite(PWM_REVERSE_FRONT_PIN, 0);
        analogWrite(PWM_REVERSE_BACK_PIN, 0);
    }
    void setPower(float val)
    {
        if (val > MAX_POWER)
            val = MAX_POWER;

        if (val < -MAX_POWER)
            val = -MAX_POWER;

        this->currentPower = val;
        unsigned char pwr = 0;

        if (val > 0)
        {
            pwr = (unsigned char)(255 * val);
            analogWrite(PWM_DIRECT_FRONT_PIN, pwr);
            analogWrite(PWM_DIRECT_BACK_PIN, pwr);
        }
        else
        {
            pwr = (unsigned char)(-255 * val);
            analogWrite(PWM_REVERSE_FRONT_PIN, pwr);
            analogWrite(PWM_REVERSE_BACK_PIN, pwr);
        }
    }

    float getCurrentPower() {
        return this->currentPower;
    }
};

#include "engine_power.h"

#endif