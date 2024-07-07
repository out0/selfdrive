#! /bin/sh

# ajustar para o caminho do avrdude / instalação da Arduino IDE no linux
AVR_DUDE=~/.arduino15/packages/arduino/tools/avrdude/6.3.0-arduino17


$AVR_DUDE/bin/avrdude "-C$AVR_DUDE/etc/avrdude.conf"  -v -V -patmega2560 -carduino "-P/dev/ttyUSB0" -b115200 -D "-Uflash:w:./arduino_controller.ino.hex:i"
