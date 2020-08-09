cmake_minimum_required(VERSION 3.2.0)

find_package(OpenCL REQUIRED)

option(BUILD_TEST "build test" ON)
option(BUILD_EXAMPLE "build example" ON)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(SRC_FILES ocl_driver.cc functors.cc)
foreach(file_path ${SRC_FILES})
    set(SRC_FILE_PATHS ${SRC_FILE_PATHS} ${CMAKE_CURRENT_LIST_DIR}/opencl/${file_path})
endforeach()

add_library(ocl_driver ${SRC_FILE_PATHS})
target_link_libraries(ocl_driver glog::glog ${OpenCL_LIBRARIES})

if(BUILD_EXAMPLE)
    add_executable(ocl_driver_demo ${CMAKE_CURRENT_LIST_DIR}/examples/ocl_driver_demo.cc)
    target_link_libraries(ocl_driver_demo ocl_driver)
endif()

if(BUILD_TEST)
    find_package(GTest REQUIRED)
    # add all unit test files
    file(GLOB_RECURSE TEST_FILES ${CMAKE_CURRENT_LIST_DIR}/opencl/*_test.cc)

    # add test helper files
    set(TEST_FILES ${TEST_FILES} ${CMAKE_CURRENT_LIST_DIR}/opencl/test_main.cc
        ${CMAKE_CURRENT_LIST_DIR}/opencl/test.cc)
    add_executable(test_main ${TEST_FILES})
    target_link_libraries(test_main ${GTEST_LIBRARIES} ocl_driver)
endif()
