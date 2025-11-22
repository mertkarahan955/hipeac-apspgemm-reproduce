cmake_minimum_required(VERSION 3.10)
project(ApSpGEMM LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/GPU
    ${CUDA_INCLUDE_DIRS}
)

# CPU source files (compile as C++)
set(CPU_SOURCES
    CSR.cpp
    COO.cpp
    NNZs.cpp
    Gustavson.cpp
)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80")

# CPU compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Create static library for CPU functions
add_library(apspgemm_cpu STATIC ${CPU_SOURCES})

# Main executable
add_executable(main main.cu)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main apspgemm_cpu ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})

# Optional: Individual test executables
add_executable(gustavson_cpu_test Gustavson.cpp CSR.cpp COO.cpp)
target_compile_definitions(gustavson_cpu_test PRIVATE STANDALONE_TEST)

message(STATUS "ApSpGEMM GPU Porting Project")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
