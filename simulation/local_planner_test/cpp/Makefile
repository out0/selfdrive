all:
	rm -rf build
	mkdir build
	cd build && cmake .. && make
	mv build/test_executor .
	rm -rf build

link:
	cd build && ln -s ../scenarios scenarios