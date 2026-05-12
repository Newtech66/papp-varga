cmake_minimum_required(VERSION 3.28)

project(bootstrap-solver LANGUAGES CXX)

add_executable(bootstrap-solver)

set(CMAKE_MODULE_PATH "cmake/" "${CMAKE_MODULE_PATH}")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# find_package(MPFR 4.0.0 REQUIRED)
# find_package(Boost REQUIRED)

find_package(OpenMP COMPONENTS CXX)
if(${OpenMP_FOUND})
    message(STATUS "Will compile with OpenMP")
    target_compile_options(bootstrap-solver PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(bootstrap-solver PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(bootstrap-solver PRIVATE cxx_std_20)
target_compile_options(bootstrap-solver PRIVATE -march=native)
target_compile_options(bootstrap-solver PRIVATE -mtune=native)
target_compile_options(bootstrap-solver PRIVATE -Wall)
target_compile_options(bootstrap-solver PRIVATE -O3)
target_compile_options(bootstrap-solver PRIVATE -fdiagnostics-color=always)

target_link_libraries(bootstrap-solver PRIVATE Eigen3::Eigen)
target_link_libraries(bootstrap-solver PRIVATE BLAS::BLAS)
target_link_libraries(bootstrap-solver PRIVATE LAPACK::LAPACK)
# target_include_directories(bootstrap-solver PRIVATE ${Boost_INCLUDE_DIRS})
# target_link_libraries(bootstrap-solver PRIVATE ${MPFR_LIBRARIES})

add_subdirectory(include)
add_subdirectory(src)
