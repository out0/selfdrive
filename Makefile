install:
	cd libdriveless && make install
	cd libgpd && make install
	cd libfastrrt && make install
	cd carla_driver && make install
	cd decision && make install

