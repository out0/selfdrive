all:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	cd build && make
	mv build/manual_control .
	rm -rf build

clean:
	rm -rf ./manual_control