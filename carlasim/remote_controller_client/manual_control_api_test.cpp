#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <string>
#include <signal.h>
#include <memory>
#include <datalink.h>

#define BROKER_IP "10.0.1.5"
#define BROKER_PORT 1883

char menu(int steering_angle, int engine_power)
{
    std::string hs;
    std::string ms;
    std::string ack;

    if (steering_angle < 0)
        hs = "[-]";
    else
        hs = "[+]";

    if (engine_power < 0)
        ms = "[-]";
    else
        ms = "[+]";

    // system("clear");
    printf("Hardware testing menu\n\n");
    printf("                   (w)       forward pwr++\n");
    printf("left pwr++   (a)   (q)   (d)      right pwr++\n");
    printf("                   (s)       backward pwr++\n");
    printf("\n\n");

    printf(">>>> moving wheeldrive: H: %s, power: %d\n", ms.c_str(), engine_power);
    printf("<--> heading: H: %s, angle: %d\n", hs.c_str(), steering_angle);

    printf("\n\n");
    printf("(r) reset\n");
    printf("(q) stop\n");
    printf("(p) AUTONOMOUS MODE ON\n");
    printf("(Esc) to quit\n\n");
    return getchar();
}

unsigned int setupTerminal()
{
    static struct termios term_flags;
    tcgetattr(STDIN_FILENO, &term_flags);
    unsigned int oldFlags = term_flags.c_lflag;
    // newt = oldt;
    term_flags.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &term_flags);
    return oldFlags;
}

void restoreTerminal(int oldFlags)
{
    static struct termios term_flags;
    tcgetattr(STDIN_FILENO, &term_flags);
    term_flags.c_lflag = oldFlags;
    tcsetattr(STDIN_FILENO, TCSANOW, &term_flags);
}

void cancelHandler(int s)
{
}

void setAuto(Datalink *link, bool value)
{
    float *data = new float[2]{
        1,
        value ? (float)1.0 : (float)0.0};
    long size = sizeof(float) * 2;
    link->sendData(data, size);
    delete data;
}
void setPower(Datalink *link, int value)
{
    float *data = new float[2]{
        2,
        (float)value / 100};
    long size = sizeof(float) * 2;
    link->sendData(data, size);
    delete data;
}
void setBrake(Datalink *link)
{
    float *data = new float[2]{
        3,
        1.0};
    long size = sizeof(float) * 2;
    link->sendData(data, size);
    delete data;
}
void setSteering(Datalink *link, int value)
{
    float *data = new float[2]{
        4,
        (float)value};
    long size = sizeof(float) * 2;
    link->sendData(data, size);
    delete data;
}
void restartMission(Datalink *link)
{
    float *data = new float[1]{
        5};
    long size = sizeof(float) * 1;
    link->sendData(data, size);
    delete data;
}

void turnAutoRunOff(bool condition, Datalink *link)
{
    if (!condition)
        return;
    setAuto(link, false);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

int main(int argc, char *argv[])
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = cancelHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    Datalink carControllerLink("10.0.2.4", 19992, 500);

    printf("waiting for the server to go online...\n");
    carControllerLink.waitReady(-1);

    auto flags = setupTerminal();
    bool run = true;

    int lastSteering = 0;
    int lastPower = 0;
    bool autoRun = false;

    while (run)
    {
        switch (menu(lastSteering, lastPower))
        {
        case 'w':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastPower += 10;
            setPower(&carControllerLink, lastPower);
            break;
        case 's':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastPower -= 10;
            setPower(&carControllerLink, lastPower);
            break;
        case 'a':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastSteering -= 10;
            setSteering(&carControllerLink, lastSteering);
            break;
        case 'd':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastSteering += 10;
            setSteering(&carControllerLink, lastSteering);
            break;
        case 'q':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastSteering = 0;
            lastPower = 0;
            setPower(&carControllerLink, lastPower);
            sleep(1);
            setSteering(&carControllerLink, lastSteering);
            sleep(1);
            setBrake(&carControllerLink);
            break;
        case 'r':
            turnAutoRunOff(autoRun, &carControllerLink);
            lastSteering = 0;
            lastPower = 0;
            setPower(&carControllerLink, lastPower);
            sleep(1);
            setSteering(&carControllerLink, lastSteering);
            sleep(1);
            restartMission(&carControllerLink);
            break;
        case 'p':
            autoRun = true;
            setAuto(&carControllerLink, true);
            break;
        case 27:
            run = false;
            break;
        default:
            break;
        }

        if (!run)
            break;
    }

    restoreTerminal(flags);
    return 0;
}