#ifndef H_STEERING_H
#define H_STEERING_H

#include <stdint.h>
#include <Servo.h>
#include <Arduino.h>

#define PHYSICAL_CENTER_ANGLE_FRONT 90
#define PHYSICAL_CENTER_ANGLE_BACK 76
#define MAX_STEERING 40

#define PWM_FRONT_PIN 3
#define PWM_BACK_PIN 2

class DoubleAckman
{
    Servo servoFront;
    Servo servoBack;

public:
    void initialize()
    {
        servoFront.attach(PWM_FRONT_PIN);
        servoBack.attach(PWM_BACK_PIN);
    }

    void setSteeringAngle(float angle)
    {
        if (angle > MAX_STEERING)
            angle = MAX_STEERING;

        if (angle < -MAX_STEERING)
            angle = -MAX_STEERING;

        servoFront.write(PHYSICAL_CENTER_ANGLE_FRONT + angle);
        servoBack.write(PHYSICAL_CENTER_ANGLE_BACK - angle);
    }
};

#endif