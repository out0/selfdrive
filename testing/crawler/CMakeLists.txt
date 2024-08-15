cmake_minimum_required(VERSION 3.18)
project(manual_control LANGUAGES CXX VERSION 1.0.0 DESCRIPTION "manual controller")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

include_directories(/usr/include/seriallink)

find_library(CRAWLER_LIB REQUIRED
    NAMES crawler
    HINTS /usr/lib
)

# find_library(SERIALLINK REQUIRED
#     NAMES seriallink
#     HINTS /usr/lib
# )


add_executable(${PROJECT_NAME} 
    manual_control.cpp
    # ../../crawler/ego_car.cpp
    # ../../utils/seriallink/seriallink.cpp
    # ../../utils/seriallink/serial_protocol.cpp
    # ../../utils/seriallink/serial_bus_linux.cpp
)

target_link_libraries(${PROJECT_NAME} 
    PkgConfig::deps 
    ${CRAWLER_LIB}
    ${SERIALLINK}
    ${CMAKE_THREAD_LIBS_INIT}
)