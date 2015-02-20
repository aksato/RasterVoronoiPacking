cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules) 

project(RasterVoronoiPackingConsole)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(../common)

find_package(Qt5Widgets)
find_package(Qt5Gui)

set(CMAKE_AUTOMOC ON)

find_package(CXX11)
IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

set(SRC
	mainConsole.cpp
	packingParametersParser.cpp
	../common/raster/glsweightset.cpp
	../common/raster/packingthread.cpp
	../common/raster/rasterinnerfitpolygon.cpp
	../common/raster/rasternofitpolygon.cpp
	../common/raster/rasterpackingproblem.cpp
	../common/raster/rasterpackingsolution.cpp
	../common/raster/rasterstrippackingsolver.cpp
	../common/raster/totaloverlapmap.cpp
	../common/packingproblem.cpp
)

add_executable(RasterVoronoiPackingConsole ${SRC})

qt5_use_modules(RasterVoronoiPackingConsole Core Gui)

set_target_properties(RasterVoronoiPackingConsole PROPERTIES COMPILE_DEFINITIONS "CONSOLE")
