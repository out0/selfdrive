#include "serial_bus_linux.h"
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <filesystem>
#include <sys/ioctl.h>

// Linux headers
#include <fcntl.h>   // Contains file controls like O_RDWR
#include <errno.h>   // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h>  // write(), read(), close()
#include <thread>
#include <chrono>

#define min(x, y) x < y ? x : y

void SerialBusLinux::waitBus()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
int openSerial(const char *device)
{
    if (!std::filesystem::exists(device))
    {
        fprintf(stderr, "[SerialLink] device not found at %s\n", device);
        return -1;
    }

    int fd = open(device, O_RDWR);

    struct termios tty;
    tty.c_cflag &= ~PARENB;        // Clear parity bit, disabling parity (most common)
    tty.c_cflag &= ~CSTOPB;        // Clear stop field, only one stop bit used in communication (most common)
    tty.c_cflag &= ~CSIZE;         // Clear all bits that set the data size
    tty.c_cflag |= CS8;            // 8 bits per byte (most common)
    tty.c_cflag &= ~CRTSCTS;       // Disable RTS/CTS hardware flow control (most common)
    tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;                                                        // Disable echo
    tty.c_lflag &= ~ECHOE;                                                       // Disable erasure
    tty.c_lflag &= ~ECHONL;                                                      // Disable new-line echo
    tty.c_lflag &= ~ISIG;                                                        // Disable interpretation of INTR, QUIT and SUSP
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);                                      // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL); // Disable any special handling of received bytes

    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed
    // tty.c_oflag &= ~OXTABS; // Prevent conversion of tabs to spaces (NOT PRESENT ON LINUX)
    // tty.c_oflag &= ~ONOEOT; // Prevent removal of C-d chars (0x004) in output (NOT PRESENT ON LINUX)

    tty.c_cc[VTIME] = 100; // Wait for up to 10s just to prevent indefinite locking
    tty.c_cc[VMIN] = 0;

    // Set in/out baud rate to be 38400
    cfsetispeed(&tty, B38400);
    cfsetospeed(&tty, B38400);

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
    {
        fprintf(stderr, "[SerialLink] Error %i from tcsetattr: %s\n", errno, strerror(errno));
        return -1;
    }

    return fd;
}

SerialBusLinux::SerialBusLinux(const char *device, int noDataTimeout_ms)
{
    this->device = device;
    this->timeoutStart = -1;
    this->noDataTimeout_ms = noDataTimeout_ms;
}

bool SerialBusLinux::initialize()
{
    this->linkFd = openSerial(this->device);
    waitBus();
    return this->linkFd > 0;
}

int SerialBusLinux::dataAvail()
{
    int result;

    if (ioctl(linkFd, FIONREAD, &result) == -1)
        return -1;

    return result;
}

unsigned char SerialBusLinux::readByte()
{
    char buffer[1] = {0};
    ssize_t size_bytes = read(linkFd, &buffer, 1);
    return buffer[0];
}

unsigned int SerialBusLinux::readBytesTo(char *buffer, int start, int maxSize) {
    int len = dataAvail();
    if (len <= 0) return 0;

    len = min(len, maxSize);
    ssize_t size_bytes = read(linkFd, buffer + start, len);
    return size_bytes;
}

void SerialBusLinux::writeByte(unsigned char val)
{
    write(this->linkFd, &val, 1);
}

void SerialBusLinux::writeBytes(char *buffer, int size)
{
    write(this->linkFd, buffer, size);
}

void SerialBusLinux::flush()
{
    sleep(1); // required to make flush work, for some reason
    tcflush(linkFd, TCIOFLUSH);
}

bool SerialBusLinux::isReady()
{
    return this->linkFd > 0;
}

long long timeNow()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void SerialBusLinux::rstTimeout()
{
    this->timeoutStart = timeNow();
}

bool SerialBusLinux::checkTimeout()
{
    if (this->noDataTimeout_ms <= 0)
        return false;

    if (timeoutStart <= 0)
    {
        timeoutStart = timeNow();
        return false;
    }

    if (timeNow() - timeoutStart > noDataTimeout_ms)
    {
#ifdef DEBUG
        printf("[seriallink] TIMEOUT\n");
#endif
        return true;
    }
    return false;
}
