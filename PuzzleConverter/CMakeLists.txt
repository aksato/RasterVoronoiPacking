#cmake_minimum_required(VERSION 2.8.11)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules) 

project(PuzzleConverter)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets)
find_package(Qt5Gui)

set(CMAKE_AUTOMOC ON)
  
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
	polydecomp-keil/config.cpp
	polydecomp-keil/polygon.cpp
	polydecomp-keil/point.cpp
	polydecomp-keil/edge.cpp
	polydecomp-keil/line.cpp
	polydecomp-keil/ray.cpp
	convexDecomposition.cpp
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
	polydecomp-keil/config.h
	polydecomp-keil/polygon.h
	polydecomp-keil/point.h
	polydecomp-keil/edge.h
	polydecomp-keil/line.h
	polydecomp-keil/ray.h
	)
	
add_executable(PuzzleConverter ${SRC} ${HEADERS})

target_link_libraries(PuzzleConverter Qt5::Core Qt5::Gui)