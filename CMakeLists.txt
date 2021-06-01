cmake_minimum_required(VERSION 3.19)
project(projetAnnuel)

set(CMAKE_CXX_STANDARD 14)

set(EIGEN_DIR "C:/eigen-3.4-rc1")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})


add_executable(projetAnnuel library.c library.cpp main.cpp)
