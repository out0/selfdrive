install: test
#install: 
	mkdir -p build
	cd build && cmake ..
	cd build && make -j12
	cd build && sudo make install
	rm -rf build
	cd unittests && python3 ./test_cuda_accel_rrt.py
	cd unittests && python3 ./test_cuda_accel_optim.py

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
