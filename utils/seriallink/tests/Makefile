all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	mv build/tst .
	rm -rf build

clean:
	rm -rf build tst

