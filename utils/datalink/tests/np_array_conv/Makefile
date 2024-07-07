all: 
	rm -rf build
	mkdir -p build
	cd build && cmake .. 
	cd build && make -j$(nproc)
	mv build/libsimplelib.so.0.0.1 ./libsimplelib.so
	mv build/unittest .
	rm -rf build


