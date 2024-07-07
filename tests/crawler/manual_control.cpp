#include <driveless/ego_car.h>
#include <stdio.h>

EgoCar car;

int main(int argc, char **argv)
{
    car.initialize("/dev/ttyACM0");
    if (car.setPower(0.2))
        printf("ACK received\n");
    else
        printf("ACK failed\n");

    if (car.setSteeringAngle(-30))
        printf("ACK received\n");
    else
        printf("ACK failed\n");

    if (car.stop())
        printf("ACK received\n");
    else
        printf("ACK failed\n");

    if (car.resetActuators())
        printf("ACK received\n");
    else
        printf("ACK failed\n");        
}