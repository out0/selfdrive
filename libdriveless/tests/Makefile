all: cuda

cuda:
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DDRIVELESS_CUDA_ENABLED=ON
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

test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest

