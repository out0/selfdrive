ARDUINO_HOME=/home/cristiano/Arduino/libraries

install:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	cd build && sudo make install
	rm -rf build
	mkdir -p ${ARDUINO_HOME}/SerialLink
	cp serial_protocol.cpp ${ARDUINO_HOME}/SerialLink
	cp serial_protocol.h ${ARDUINO_HOME}/SerialLink
	cp serial_bus.h ${ARDUINO_HOME}/SerialLink
	cp seriallink.cpp ${ARDUINO_HOME}/SerialLink
	cp seriallink.h ${ARDUINO_HOME}/SerialLink
	rm -rf /home/cristiano/.var/app/cc.arduino.IDE2/cache/arduino/sketches/*
