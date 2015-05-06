#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/packingthread.h"
#include "glsweightviewerdialog.h"
#include "zoomedmapviewdialog.h"
#include "runconfigurationsdialog.h"
#include <QMainWindow>
#include <memory>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void loadPuzzle();
    void loadZoomedPuzzle();
    void updateAngleComboBox(int id);
    void updatePositionValues(QPointF pos);
    void printCurrentSolution();
    void generateCurrentTotalOverlapMap();
	void generateCurrentTotalOverlapMapGPU();
	void generateCurrentTotalOverlapMapCache();
    void translateCurrentToMinimumPosition();
	void translateCurrentToMinimumPositionGPU();
    void createRandomLayout();
    void changeContainerWidth();
    void showGlobalOverlap();
    void localSearch();
	void localSearchGPU();
	void localSearchCache();
    void generateCurrentTotalGlsWeightedOverlapMap();
	void generateCurrentTotalGlsWeightedOverlapMapGPU();
    void updateGlsWeightedOverlapMap();
    void resetGlsWeightedOverlapMap();
    void translateCurrentToGlsWeightedMinimumPosition();
	void translateCurrentToGlsWeightedMinimumPositionGPU();
	void generateCurrentTotalGlsWeightedOverlapMapCache();
    void glsWeightedlocalSearch();
	void glsWeightedlocalSearchGPU();
	void glsWeightedlocalSearchCache();

    void showZoomedMap();
    void translateCurrentToMinimumZoomedPosition();
    void showZoomedGlobalOverlap();
    void updateZoomedGlsWeights();
    void zoomedlocalSearch();

    void executePacking();
	void showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length);
	void showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed, uint seed);
	void showExecutionMinLengthObtained(int minLength, int totalItNum, qreal elapsed, uint seed);

    void saveSolution();
	void saveZoomedSolution();
    void loadSolution();
    void exportSolutionToSvg();

	void printDensity();
	void printCacheInfo();
	void printGlsWeightedCacheInfo();

private:
    void createOverlapMessageBox(qreal globalOverlap, QVector<qreal> &individualOverlaps, qreal scale);

    Ui::MainWindow *ui;
    std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> rasterProblem, rasterZoomedProblem;
    RASTERVORONOIPACKING::RasterPackingSolution solution;
    std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERVORONOIPACKING::RasterStripPackingParameters params;

    ZoomedMapViewDialog zoomedMapViewer;
    GlsWeightViewerDialog weightViewer;
    RunConfigurationsDialog runConfig;
    PackingThread runThread;

    int accContainerShrink;
	qreal totalArea; qreal containerWidth;

};

#endif // MAINWINDOW_H
