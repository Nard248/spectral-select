cmake_minimum_required(VERSION 3.20)
project(lichen_exact_selector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(lichen_exact src/main.cpp)
target_compile_options(lichen_exact PRIVATE -O3 -ffast-math -mcpu=native -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(lichen_exact PRIVATE Threads::Threads)
