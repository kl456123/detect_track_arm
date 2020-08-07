cmake_minimum_required(VERSION 3.2.0)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(SRC_FILES ocl_driver.cc functors.cc)
foreach(file_path ${SRC_FILES})
    set(SRC_FILE_PATHS ${SRC_FILE_PATHS} ${CMAKE_CURRENT_LIST_DIR}/opencl/${file_path})
endforeach()

add_library(ocl_driver ${SRC_FILE_PATHS})
target_link_libraries(ocl_driver glog::glog ${OpenCL_LIBRARIES})

add_executable(ocl_driver_test ${CMAKE_CURRENT_LIST_DIR}/opencl/ocl_driver_test.cc)
target_link_libraries(ocl_driver_test ocl_driver)
