install:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	cp build/bevconv .
	rm -rf build
