cmake_minimum_required(VERSION 3.16)
project(AdvancedAlgorithms)
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
add_compile_options(-Wall -Wextra -pedantic)

set(CMAKE_CXX_STANDARD 20)

add_executable(MST secondMST.cpp)
