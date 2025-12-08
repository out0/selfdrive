install:
	cd datalink && make install
	cd libdriveless && make install
	cd libgpd && make install
	cd FastRRT && make install
	cd carla_driver && make install
	cd decision && make install

