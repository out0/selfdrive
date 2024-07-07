cmake_minimum_required(VERSION 3.11)
project(crawler LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "crawler driver")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

include_directories(/usr/include/seriallink)

SET(TARGET_H
    ego_car.h
)
 
add_library(${PROJECT_NAME} SHARED
    ego_car.cpp
    pybind.cpp
    ../utils/seriallink/serial_bus_linux.cpp
    ../utils/seriallink/serial_protocol.cpp
    ../utils/seriallink/seriallink.cpp    
)

# find_library(SERIALLINK REQUIRED
#     NAMES seriallink
#     HINTS /usr/lib
# )


target_link_libraries(${PROJECT_NAME} 
    PkgConfig::deps 
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLIB_LDFLAGS}
    ${SERIALLINK}
)

# add_executable(test_link_sender 
#     test_link_sender.cpp
#     serial_bus_linux.cpp
#     serial_link_arduino/seriallink.cpp
# )

# add_definitions(-D __SERIALLINK_COMPILE=1)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

target_sources(${PROJECT_NAME} PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    FILES 
        ego_car.h
    )

install(TARGETS 
    ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS 
    ${PROJECT_NAME} 
    FILE_SET public_headers
    DESTINATION "/usr/include/driveless"
)    