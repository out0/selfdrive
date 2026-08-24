all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)

install: bin 
	cd build && sudo make install
	mkdir -p python_bind/app/pywpmp/cpp
	cp build/libwpmp.so.0.1.0	python_bind/app/pywpmp/cpp/libwpmp.so
#	cd python_bind && pip3 install .
	cd python_bind && pip3 install . --break-system-packages	
	cd python_bind && rm -rf build
	cd python_bind && rm -rf dist	
	rm -rf build

test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest
