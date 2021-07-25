cmake_minimum_required(VERSION 3.19)
project(annualProject)

set(CMAKE_CXX_STANDARD 14)

SET( EIGEN3_INCLUDE_DIR "C:/Users/33660/Documents/ESGI/Projet_annuel/eigen-3.4-rc1/eigen-3.4-rc1")
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

add_library(annualProject SHARED library.cpp)