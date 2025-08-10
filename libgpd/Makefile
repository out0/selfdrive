all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)

local: bin
	cp build/libgpd.so.1.0.0	./libgpd.so
	rm -rf build

#install: update_header bin 
install: bin 
	cd build && sudo make install
	rm -rf python_bind/app/pygpd/cpp
	mkdir -p python_bind/app/pygpd/cpp
	cp build/libgpd.so.1.0.0	python_bind/app/pygpd/cpp/libgpd.so
	cd python_bind && pip3 install .
	cd python_bind && rm -rf build
	cd python_bind && rm -rf dist	
	rm -rf build




test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest

update_header:
	python3 ./header_upd.py