#cmake_minimum_required(VERSION 2.8.11)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules) 

project(RasterVoronoiBatchExecutor)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets)
find_package(CUDA QUIET REQUIRED)

set(CMAKE_AUTOMOC ON)

find_package(CXX11)
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

set(SRC 
	main.cpp
	packingParametersParser.cpp
	)
	
set(HEADERS
	packingParametersParser.h
	)
	
add_executable(RasterVoronoiBatchExecutor ${SRC} ${HEADERS})
qt5_use_modules(RasterVoronoiBatchExecutor Core)