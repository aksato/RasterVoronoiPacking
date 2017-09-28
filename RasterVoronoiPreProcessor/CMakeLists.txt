#cmake_minimum_required(VERSION 2.8.11)
cmake_minimum_required(VERSION 3.0)
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
	../RasterVoronoiPacking/common/packingproblem.cpp
	rasterization.cpp
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
	cluster/clusterizator.cpp
	kernel.cu
	)
	
set(HEADERS
	../RasterVoronoiPacking/common/packingproblem.h
	parametersParser.h
	colormap.h
	annealing/cContainer.h
	annealing/cPolygon.h
	annealing/cRectangularContainer.h
	annealing/cShape.h
	annealing/cShapeInlines.h
	annealing/cShapeParameters.h
	annealing/cShapePlacement.h
	annealing/cVector.h
	annealing/problemInstance.h
	polybool/ObjHeap.h
	polybool/pbarea.h
	polybool/pbdefs.h
	polybool/pbgeom.h
	polybool/pbimpl.h
	polybool/pbio.h
	polybool/pbtria.h
	polybool/polybool.h
	polybool/Sort.h
	cluster/clusterizator.h
	)
	
cuda_add_executable(RasterPackingPreProcessor ${SRC} ${HEADERS})

qt5_use_modules(RasterPackingPreProcessor Core Gui)