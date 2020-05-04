cmake_minimum_required(VERSION 3.10)
project(pdc)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
#add_compile_options(-Wall -Wextra -pedantic -Werror)
set(CMAKE_CXX_STANDARD 11)

#SET(CMAKE_C_COMPILER gcc-9)
#SET(CMAKE_CXX_COMPILER g++-9)

# Sequential
add_executable(matFact matFact.cpp)

# OpenMP
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
add_executable(matFact-omp matFact-omp.cpp)
set_target_properties(matFact-omp PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")

# MPI
add_executable(matFact-mpi matFact-mpi.cpp)
find_package(MPI REQUIRED)
target_link_libraries(matFact-mpi ${MPI_LIBRARIES})
include_directories(${MPI_INCLUDE_PATH})
set_target_properties(matFact-mpi PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
