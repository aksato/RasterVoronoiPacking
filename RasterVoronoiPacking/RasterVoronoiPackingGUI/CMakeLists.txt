#cmake_minimum_required(VERSION 2.8.11)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules) 

project(RasterVoronoiPackingGUI)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(../common)

find_package(Qt5Widgets)
find_package(Qt5Gui)

set(CMAKE_AUTOMOC ON)

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "-std=c++11")
endif()

set(SRC 
	colormap.cpp 
	glsweightviewerdialog.cpp 
	mainwindow.cpp 
	../common/packingproblem.cpp
	runconfigurationsdialog.cpp
	zoomedmapviewdialog.cpp
	main.cpp
	../common/raster/totaloverlapmap.cpp
	../common/raster/rasterstrippackingsolver.cpp
	../common/raster/rasterstrippackingsolvergls.cpp
	../common/raster/rasterstrippackingsolverdoublegls.cpp
	../common/raster/rasterpackingsolution.cpp
	../common/raster/rasterpackingproblem.cpp
	../common/raster/rasternofitpolygon.cpp
	../common/raster/rasterinnerfitpolygon.cpp
	../common/raster/packingthread.cpp
	../common/raster/packingclusterthread.cpp
	../common/raster/glsweightset.cpp
	viewer/graphicsindexedrectitem.cpp
	viewer/packingitem.cpp
	viewer/packingviewer.cpp
	viewer/weightviewer.cpp
	viewer/zoomedmapview.cpp
	)
	
set(HEADERS
	colormap.h 
	glsweightviewerdialog.h 
	mainwindow.h 
	../common/packingproblem.h
	runconfigurationsdialog.h
	zoomedmapviewdialog.h
	../common/raster/totaloverlapmap.h
	../common/raster/rasterstrippackingsolver.h
	../common/raster/rasterstrippackingsolvergls.h
	../common/raster/rasterstrippackingsolverdoublegls.h
	../common/raster/rasterpackingsolution.h
	../common/raster/rasterstrippackingparameters.h
	../common/raster/rasterpackingproblem.h
	../common/raster/rasternofitpolygon.h
	../common/raster/rasterinnerfitpolygon.h
	../common/raster/packingthread.h
	../common/raster/packingclusterthread.h
	../common/raster/glsweightset.h
	viewer/graphicsindexedrectitem.h
	viewer/packingitem.h
	viewer/packingviewer.h
	viewer/weightviewer.h
	viewer/zoomedmapview.h
	)

qt5_wrap_ui(UIS_HDRS
    glsweightviewerdialog.ui
    mainwindow.ui
    packingviewer.ui
    runconfigurationsdialog.ui
	zoomedmapviewdialog.ui
)

qt5_add_resources (ResourceSources resources.qrc)
	
add_executable(RasterVoronoiPackingGUI ${SRC} ${HEADERS} ${UIS_HDRS} ${ResourceSources})

qt5_use_modules(RasterVoronoiPackingGUI Widgets Gui Svg)