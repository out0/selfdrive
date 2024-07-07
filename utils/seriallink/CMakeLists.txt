cmake_minimum_required(VERSION 3.11)
project(seriallink LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "seriallink")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)


SET(TARGET_H
    serial_bus_linux.h
    serial_protocol.h
    seriallink.h
    serial_bus.h
)
 
add_library(${PROJECT_NAME} SHARED
    serial_bus_linux.cpp
    serial_protocol.cpp
    seriallink.cpp
)

# target_link_libraries(seriallink 
#         PkgConfig::deps 
#         ${CMAKE_THREAD_LIBS_INIT}
#         ${GLIB_LDFLAGS}
       
# )

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
        serial_bus_linux.h
        serial_protocol.h
        serial_bus.h
        seriallink.h
    )

install(TARGETS 
    ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS 
    ${PROJECT_NAME} 
    FILE_SET public_headers
    DESTINATION "/usr/include/seriallink"
)    