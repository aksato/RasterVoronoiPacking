cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules) 

project(RasterVoronoiPacking)

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
	colormap.cpp 
	glsweightviewerdialog.cpp 
	mainwindow.cpp 
	overlapmap.cpp
	packingproblem.cpp
	runconfigurationsdialog.cpp
	zoomedmapviewdialog.cpp
	main.cpp
	raster/totaloverlapmap.cpp
	raster/rasterstrippackingsolver.cpp
	raster/rasterpackingsolution.cpp
	raster/rasterpackingproblem.cpp
	raster/rasternofitpolygon.cpp
	raster/rasterinnerfitpolygon.cpp
	raster/packingthread.cpp
	raster/glsweightset.cpp
	viewer/graphicsindexedrectitem.cpp
	viewer/packingitem.cpp
	viewer/packingviewer.cpp
	viewer/weightviewer.cpp
	viewer/zoomedmapview.cpp)
	
qt5_wrap_ui(UIS_HDRS
    glsweightviewerdialog.ui
    mainwindow.ui
    packingviewer.ui
    runconfigurationsdialog.ui
	zoomedmapviewdialog.ui
)

qt5_add_resources (ResourceSources resources.qrc)
	
add_executable(RasterVoronoiPacking ${SRC} ${UIS_HDRS} ${ResourceSources})

qt5_use_modules(RasterVoronoiPacking Widgets Gui)