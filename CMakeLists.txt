cmake_minimum_required(VERSION 3.28)

project(papp-varga LANGUAGES CXX)

add_executable(papp-varga)

set(CMAKE_MODULE_PATH "cmake/" "${CMAKE_MODULE_PATH}")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# find_package(MPFR 4.0.0 REQUIRED)
find_package(Boost REQUIRED)

find_package(OpenMP COMPONENTS CXX)
if(${OpenMP_FOUND})
    message(STATUS "Will compile with OpenMP")
    target_compile_options(papp-varga PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(papp-varga PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(papp-varga PRIVATE cxx_std_17)
target_compile_options(papp-varga PRIVATE -march=native)
target_compile_options(papp-varga PRIVATE -O3)
target_compile_options(papp-varga PRIVATE -fmax-errors=1)
target_compile_options(papp-varga PRIVATE -DNDEBUG)

target_link_libraries(papp-varga PRIVATE Eigen3::Eigen)
target_link_libraries(papp-varga PRIVATE BLAS::BLAS)
target_link_libraries(papp-varga PRIVATE LAPACK::LAPACK)
target_include_directories(papp-varga PRIVATE ${Boost_INCLUDE_DIRS})
# target_link_libraries(papp-varga PRIVATE ${MPFR_LIBRARIES})

add_subdirectory(include)
add_subdirectory(src)
