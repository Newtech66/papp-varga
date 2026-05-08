cmake_minimum_required(VERSION 3.28)

project(papp-varga LANGUAGES CXX)

set(CMAKE_MODULE_PATH "cmake/" "${CMAKE_MODULE_PATH}")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
# find_package(MPFR 4.0.0 REQUIRED)
find_package(Boost REQUIRED)

add_executable(papp-varga)

target_compile_features(papp-varga PRIVATE cxx_std_17)
target_compile_options(papp-varga PRIVATE -march=native)
target_compile_options(papp-varga PRIVATE -O2)
target_compile_options(papp-varga PRIVATE -fmax-errors=1)

# Need ways to set -DNDEBUG, -fopenmp

target_link_libraries(papp-varga PRIVATE Eigen3::Eigen)
target_link_libraries(papp-varga PRIVATE BLAS::BLAS)
target_include_directories(papp-varga PRIVATE ${Boost_INCLUDE_DIRS})
# target_link_libraries(papp-varga PRIVATE ${MPFR_LIBRARIES})

add_subdirectory(include)
add_subdirectory(src)
