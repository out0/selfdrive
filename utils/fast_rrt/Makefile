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
	rm -rf build


test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest
