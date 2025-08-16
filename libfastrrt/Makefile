all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)

local: bin
	cp build/libfastrrt.so.1	./libfastrrt.so
	rm -rf build

install: bin 
	cd build && make install
	cp build/libfastrrt.so.0.1.0	python_bind/app/pyfastrrt/cpp/libfastrrt.so
	cd python_bind && pip3 install .
	cd python_bind && rm -rf build
	cd python_bind && rm -rf dist	
	rm -rf build

test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest
