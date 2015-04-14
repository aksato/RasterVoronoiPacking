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
	void showExecutionStatus(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, int curLength, int minLength);
	void showExecutionFinishedStatus(int totalItNum, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, int minLength);

    void saveSolution();
    void loadSolution();
    void exportSolutionToSvg();

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

};

#endif // MAINWINDOW_H
