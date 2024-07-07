cmake_minimum_required(VERSION 3.18)
project(crawler LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "full crawler driver")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

include_directories(../../../utils/seriallink)
 
add_library(${PROJECT_NAME} SHARED
    ../../../crawler/ego_car.cpp
    ../../../utils/seriallink/serial_bus_linux.cpp
    ../../../utils/seriallink/serial_protocol.cpp
    ../../../utils/seriallink/seriallink.cpp
    ../../../crawler/pybind.cpp
)


target_link_libraries(${PROJECT_NAME} 
    PkgConfig::deps 
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLIB_LDFLAGS}
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
