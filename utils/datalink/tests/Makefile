all: 
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug
	cd build && make -j$(nproc) VERBOSE=1
	mv build/unittest .
	mv build/tst_client .
	mv build/tst_server .
	mv build/tst_bridge .
	rm -rf build

test: all
	./unittest
	rm unittest

