all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)

local: bin
	cp build/libdriveless.so.2.0.0	./libdriveless.so
	rm -rf build

#install: update_header bin 
install: bin 
	cd build && sudo make install
	rm -rf python_bind/app/pydriveless/cpp
	mkdir -p python_bind/app/pydriveless/cpp
	cp build/libdriveless.so.2.0.0	python_bind/app/pydriveless/cpp/libdriveless.so
	cd python_bind && pip3 install .
	cd python_bind && rm -rf build
	cd python_bind && rm -rf dist	
	rm -rf build


clean:
	rm -rf build unittest

update_header:
	python3 ./header_upd.py