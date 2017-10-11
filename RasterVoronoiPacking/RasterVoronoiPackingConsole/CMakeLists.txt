#cmake_minimum_required(VERSION 2.8.11)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules) 

project(RasterVoronoiPackingConsole)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(../common)

find_package(Qt5Widgets)
find_package(Qt5Gui)
find_package(Qt5Svg)

set(CMAKE_AUTOMOC ON)

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

set(SRC
	mainConsole.cpp
	consolepackingloader.cpp
	packingParametersParser.cpp
	../common/packingproblem.cpp
	../common/raster/totaloverlapmap.cpp
	../common/raster/rasterstrippackingsolver.cpp
	../common/raster/rasterstrippackingsolver2d.cpp
	../common/raster/rasteroverlapevaluator.cpp
	../common/raster/rasteroverlapevaluatorgls.cpp
	../common/raster/rasteroverlapevaluatordoublegls.cpp
	../common/raster/rasterpackingsolution.cpp
	../common/raster/rasterpackingproblem.cpp
	../common/raster/rasternofitpolygon.cpp
	../common/raster/rasterinnerfitpolygon.cpp
	../common/raster/packingthread.cpp
	../common/raster/packing2dthread.cpp
	../common/raster/packingclusterthread.cpp
	../common/raster/glsweightset.cpp
	)
	
set(HEADERS
	consolepackingloader.h
	packingParametersParser.h
	../common/packingproblem.h
	../common/raster/totaloverlapmap.h
	../common/raster/rasterstrippackingsolver.h
	../common/raster/rasteroverlapevaluator.h
	../common/raster/rasterpackingsolution.h
	../common/raster/rasterstrippackingparameters.h
	../common/raster/rasterpackingproblem.h
	../common/raster/rasternofitpolygon.h
	../common/raster/rasterinnerfitpolygon.h
	../common/raster/packingthread.h
	../common/raster/packing2dthread.h
	../common/raster/packingclusterthread.h
	../common/raster/glsweightset.h
	)

add_executable(RasterVoronoiPackingConsole ${SRC} ${HEADERS})

target_link_libraries(RasterVoronoiPackingConsole Qt5::Core Qt5::Gui Qt5::Svg)

set_target_properties(RasterVoronoiPackingConsole PROPERTIES COMPILE_DEFINITIONS "CONSOLE")
