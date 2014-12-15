cmake_minimum_required(VERSION 2.8.11)

project(RasterVoronoiPacking)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Find the QtWidgets library
find_package(Qt5Widgets)

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
	viewer/weightviewer.cpp
	viewer/zoomedmapview.cpp)
	
QT5_WRAP_UI(UIS_HDRS
    glsweightviewerdialog.ui
    mainwindow.ui
    packingviewer.ui
    runconfigurationsdialog.ui
	zoomedmapviewdialog.ui
)
	
	

add_executable(RasterVoronoiPacking ${SRC} ${UIS_HDRS})

target_link_libraries(RasterVoronoiPacking Qt5::Widgets)