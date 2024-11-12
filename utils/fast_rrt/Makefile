install: test
#install: 
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	cd build && sudo make install
	rm -rf build

test:
	cd unittests && make test
	cd unittests && python3 ./test_curve_gen.py

ignore:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	cd build && sudo make install
	rm -rf build

jetson:
	mkdir -p build
	cd build && cmake .. -DCUDA_LIB_JETSON="true"
	cd build && make -j$(nproc)
	cd build && sudo make install
	rm -rf build
