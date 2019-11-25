
#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(test C CXX)

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4.mnn include
include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN/include)

# mnn lib
add_library( MNN SHARED IMPORTED )
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
add_executable(detector ${CMAKE_CURRENT_LIST_DIR}/src/detect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common.cpp)

add_executable(tracker ${CMAKE_CURRENT_LIST_DIR}/src/track.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/common.cpp)

target_link_libraries(detector
                      MNN
                      ${OpenCV_LIBS} -ldl
					  )
target_link_libraries(tracker
                      MNN
                      ${OpenCV_LIBS} -ldl
					  )
