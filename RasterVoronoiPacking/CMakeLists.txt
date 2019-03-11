cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(RasterVoronoiPacking)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Svg REQUIRED)
find_package(Eigen3 3.3.5 REQUIRED)  # We only need this version for CUDA
find_package(CUDA 10 REQUIRED)

set(CMAKE_AUTOMOC ON)
include_directories(${Qt5Widgets_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} common)

set(common_SRCS
	common/colormap.cpp 
	common/packingproblem.cpp
	common/raster/totaloverlapmap.cpp
	common/raster/totaloverlapmapcache.cpp
	common/raster/totaloverlapmatrix.cpp
	common/raster/rasterstrippackingcompactor.cpp
	common/raster/rastersquarepackingcompactor.cpp
	common/raster/rasterrectpackingcompactor.cpp
	common/raster/rasterstrippackingsolver.cpp
	common/raster/rasteroverlapevaluator.cpp
	common/raster/rasteroverlapevaluatorfull.cpp
	common/raster/rasteroverlapevaluatorgls.cpp
	common/raster/rasteroverlapevaluatormatrixgls.cpp
	common/raster/rasteroverlapevaluatordoublegls.cpp
	common/raster/rasterpackingsolution.cpp
	common/raster/rasterpackingproblem.cpp
	common/raster/rasternofitpolygon.cpp
	common/raster/rasterinnerfitpolygon.cpp
	common/raster/glsweightset.cpp
	)

set(console_SRCS
	RasterVoronoiPackingConsole/consolepackingloader.cpp
	common/raster/packingthread.cpp
	)
	
set(cuda_SRCS
	common/cuda/rasteroverlapevaluatorcudagls.cu
	common/cuda/rasteroverlapevaluatormatrixgls.cu
	common/cuda/rasterpackingcudaproblem.cu
	common/cuda/totaloverlapmapcuda.cu
	common/cuda/totaloverlapmatrix.cu
	common/cuda/glsweightsetcuda.cu
	)

set(gui_SRCS
	common/raster/packingthread.cpp
	RasterVoronoiPackingGUI/glsweightviewerdialog.cpp 
	RasterVoronoiPackingGUI/main.cpp
	RasterVoronoiPackingGUI/mainwindow.cpp 
	RasterVoronoiPackingGUI/runconfigurationsdialog.cpp
	RasterVoronoiPackingGUI/zoomedmapviewdialog.cpp
	RasterVoronoiPackingGUI/viewer/graphicsindexedrectitem.cpp
	RasterVoronoiPackingGUI/viewer/packingitem.cpp
	RasterVoronoiPackingGUI/viewer/packingviewer.cpp
	RasterVoronoiPackingGUI/viewer/weightviewer.cpp
	RasterVoronoiPackingGUI/viewer/zoomedmapview.cpp
	)	
	
qt5_wrap_ui(UIS_HDRS
    RasterVoronoiPackingGUI/glsweightviewerdialog.ui
    RasterVoronoiPackingGUI/mainwindow.ui
    RasterVoronoiPackingGUI/packingviewer.ui
    RasterVoronoiPackingGUI/runconfigurationsdialog.ui
	RasterVoronoiPackingGUI/zoomedmapviewdialog.ui
)

qt5_add_resources (ResourceSources RasterVoronoiPackingGUI/resources.qrc)

add_library(packing_core OBJECT ${common_SRCS})

add_executable(RasterVoronoiPackingGUI RasterVoronoiPackingGUI/main.cpp ${gui_SRCS} ${UIS_HDRS} ${ResourceSources} $<TARGET_OBJECTS:packing_core>)
target_link_libraries(RasterVoronoiPackingGUI Qt5::Core Qt5::Gui Qt5::Svg)

add_executable(RasterVoronoiPackingConsole RasterVoronoiPackingConsole/mainConsole.cpp ${console_SRCS} $<TARGET_OBJECTS:packing_core>)
target_link_libraries(RasterVoronoiPackingConsole Qt5::Core Qt5::Gui Qt5::Svg)
set_target_properties(RasterVoronoiPackingConsole PROPERTIES COMPILE_DEFINITIONS "CONSOLE")
set_property(TARGET RasterVoronoiPackingConsole PROPERTY CXX_STANDARD 17)
set_property(TARGET RasterVoronoiPackingConsole PROPERTY CXX_STANDARD_REQUIRED ON)

cuda_add_executable(RasterVoronoiPackingCudaTester RasterVoronoiPackingConsole/mainCudaTest.cu ${cuda_SRCS} $<TARGET_OBJECTS:packing_core>)
target_link_libraries(RasterVoronoiPackingCudaTester Qt5::Core Qt5::Gui Qt5::Svg)
target_compile_options(RasterVoronoiPackingCudaTester PRIVATE /arch:AVX2)
set_property(TARGET RasterVoronoiPackingConsole PROPERTY CXX_STANDARD 17)
set_property(TARGET RasterVoronoiPackingConsole PROPERTY CXX_STANDARD_REQUIRED ON)