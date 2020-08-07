
#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(test C CXX)

option(BUILD_SDK "Enable stereo camera" OFF)
option(BUILD_TRACKER "Build tracker" OFF)

#opencv
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glog REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4.mnn include
include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/include)

#5. opencl driver wrapper headers
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/opencl/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/lib)


# mnn lib
add_library( MNN SHARED IMPORTED )
if (BUILD_SDK)
    add_definitions(-DUSE_SDK)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/src/sdk)
    add_subdirectory(./src/sdk)
endif()
set_target_properties(
    MNN
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/MNN/lib/libMNN.so
    )

#6.source directory
# file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h
# ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
# set(TEST_COMPILE_CODE ${TEST_SRC})


# add_executable(test ${TEST_COMPILE_CODE})


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/opencl)

set(DETECTOR_CPP)
list(APPEND DETECTOR_CPP
    ${CMAKE_CURRENT_LIST_DIR}/src/detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/centernet_detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/instance_manager.cpp
    )


set(DETECTOR_DEPEND MNN)
if (BUILD_SDK)
    list(APPEND DETECTOR_DEPEND SDK)
endif()

add_executable(detector ${DETECTOR_CPP} ${CMAKE_CURRENT_LIST_DIR}/src/detect.cpp)
target_link_libraries(detector
    ${DETECTOR_DEPEND}
    ${OpenCV_LIBS} -ldl
    ocl_driver
    )

add_executable(test_detector ${DETECTOR_CPP} ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)
target_link_libraries(test_detector
    ${DETECTOR_DEPEND}
    ${OpenCV_LIBS} -ldl
    ocl_driver
    )

if(BUILD_TRACKER)
    add_executable(tracker ${CMAKE_CURRENT_LIST_DIR}/src/track.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/tracker.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/common.cpp)

    target_link_libraries(tracker
        MNN
        ${OpenCV_LIBS} -ldl
        )
endif()
