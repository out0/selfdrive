install:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	mv build/libcrawler.so.1.0.0 ./libcrawler.so
	rm -rf build
