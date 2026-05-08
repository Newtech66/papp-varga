cmake_minimum_required(VERSION 3.28)

project(papp-varga LANGUAGES CXX)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)

add_executable(papp-varga)

target_compile_features(papp-varga PRIVATE cxx_std_17)
target_compile_options(papp-varga PRIVATE -march=native)
target_compile_options(papp-varga PRIVATE -O2)

# Need ways to set -DNDEBUG, -fopenmp

target_link_libraries(papp-varga PRIVATE Eigen3::Eigen)
target_link_libraries(papp-varga PRIVATE BLAS::BLAS)

target_sources(papp-varga
PRIVATE
FILE_SET templateHeaders
TYPE HEADERS
BASE_DIRS include
FILES
include/cones.hpp
include/linsolver.hpp
include/common_typedefs.hpp
include/complex_deduction.hpp

PRIVATE
FILE_SET coneHeaders
TYPE HEADERS
BASE_DIRS cones
FILES
cones/logperspecepi.cpp
cones/possemidefinite.cpp

PRIVATE
FILE_SET otherHeaders
TYPE HEADERS
FILES
solver.cpp
vectorize.cpp)

target_sources(papp-varga
PRIVATE
src/cones.tpp
src/linsolver.tpp
src/model.cpp
src/point.cpp
src/reader.cpp
papp_varga.cpp
)
