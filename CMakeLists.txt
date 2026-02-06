cmake_minimum_required(VERSION 3.10)
project(RBC_Seg)

# set(OpenCV_DIR "E:/opencv/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(rbc_seg rbc_segmentation.cpp)
target_link_libraries(rbc_seg ${OpenCV_LIBS})
