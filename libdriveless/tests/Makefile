all: cuda

cuda:
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DDRIVELESS_CUDA_ENABLED=ON -DCMAKE_BUILD_TYPE=Debug
	cd build && make -j$(nproc)
	mv build/unittest .
	rm -rf build

cpu:
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DDRIVELESS_CUDA_ENABLED=OFF 
	cd build && make -j$(nproc)
	mv build/unittest .
	rm -rf build

test: cuda
	./unittest
	rm unittest

cputest: cpu
	./unittest
	rm unittest

clean:
	rm -rf build unittest

