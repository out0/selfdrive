all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	mv build/unittest .
	rm -rf build

test: bin
	./unittest
	rm unittest
	rm test_file.mkv

clean:
	rm -rf build unittest

