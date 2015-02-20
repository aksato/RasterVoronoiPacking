cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules) 

project(RasterVoronoiConsole)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

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
	../raster/glsweightset.cpp
	../raster/packingthread.cpp
	../raster/rasterinnerfitpolygon.cpp
	../raster/rasternofitpolygon.cpp
	../raster/rasterpackingproblem.cpp
	../raster/rasterpackingsolution.cpp
	../raster/rasterstrippackingsolver.cpp
	../raster/totaloverlapmap.cpp
	../packingproblem.cpp
)

add_executable(RasterVoronoiConsole ${SRC})

qt5_use_modules(RasterVoronoiConsole Core Gui)

set_target_properties(RasterVoronoiConsole PROPERTIES COMPILE_DEFINITIONS "CONSOLE")
