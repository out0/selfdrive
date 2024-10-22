install: test
#install: 
	mkdir -p build
	cd build && cmake ..
	cd build && make -j12
	cd build && sudo make install
	rm -rf build

test:
	cd unittests && make test

ignore:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j12
	cd build && sudo make install
	rm -rf build

jetson:
	mkdir -p build
	cd build && cmake .. -DCUDA_LIB_JETSON="true"
	cd build && make -j6
	cd build && sudo make install
	rm -rf build
