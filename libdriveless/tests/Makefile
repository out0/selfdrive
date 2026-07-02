all: cuda

cuda:
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DCUDA_ENABLE=ON
	cd build && make -j$(nproc)
	mv build/unittest .
	rm -rf build

cpu:
	rm -rf build
	mkdir -p build
	cd build && cmake .. -DCUDA_ENABLE=OFF
	cd build && make -j$(nproc)
	mv build/unittest .
	rm -rf build

test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest

