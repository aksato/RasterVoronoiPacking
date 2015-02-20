cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules) 

project(RasterPackingPreProcessor)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_22,code=sm_22)

set(CMAKE_AUTOMOC ON)

find_package(CXX11)
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

set(SRC 
	main.cpp
	PackingProblem.cpp 
	parametersParser.cpp 
	annealing/cPolygon.cpp
	annealing/cRectangularContainer.cpp
	annealing/cShape.cpp
	annealing/problemInstance.cpp
	polybool/pbgeom.cpp
	polybool/pbio.cpp
	polybool/pbpolys.cpp
	polybool/pbsweep.cpp
	polybool/polybool.cpp
	polybool/triacons.cpp
	polybool/triamono.cpp
	kernel.cu
	)
	
cuda_add_executable(RasterPackingPreProcessor ${SRC})

qt5_use_modules(RasterPackingPreProcessor Core Gui)