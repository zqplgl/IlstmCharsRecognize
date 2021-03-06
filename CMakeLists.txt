cmake_minimum_required(VERSION 3.5)

project(lstmCharsRecognize)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR})

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/ctc)

message(STATUS ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ctc/include)
include_directories(/usr/local/cuda/include)
file(GLOB SRC src/*.cpp)
find_package(OpenCV 3.4 REQUIRED)

find_library(Boostpython_LIB boost_python3 REQUIRED)
message(STATUS Found Boostpython_LIB:  ${Boostpython_LIB})
find_library(Boostsystem_LIB boost_system REQUIRED)
message(STATUS Found Boostsystem_LIB:  ${Boostsystem_LIB})
find_library(Glog_LIB glog REQUIRED)
message(STATUS Found Glog_LIB:  ${Glog_LIB})
find_library(Caffe_LIB caffe REQUIRED)
message(STATUS Found Caffe_LIB:  ${Caffe_LIB})

add_library(IlstmCharsRecognize SHARED ${SRC})
target_link_libraries(IlstmCharsRecognize ${OpenCV_LIBS} ${Boostpython_LIB} ${Boostsystem_LIB} ${Glog_LIB} ${Caffe_LIB} ctc)
add_executable(lstmCharsRecognize ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(lstmCharsRecognize ${OpenCV_LIBS} IlstmCharsRecognize)
