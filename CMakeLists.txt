cmake_minimum_required(VERSION 3.8)
project(adaptive_bw_bench VERSION 0.1.0 LANGUAGES CXX C)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# require pthreads
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add the executable here 
add_executable(adaptive_bw_bench adaptive_bw_bench.c)
target_link_libraries(adaptive_bw_bench unstickymem Threads::Threads numa)
