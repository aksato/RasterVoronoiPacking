#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/packingthread.h"
#include "raster/packing2dthread.h"
#include "raster/packingclusterthread.h"
#include "glsweightviewerdialog.h"
#include "zoomedmapviewdialog.h"
#include "runconfigurationsdialog.h"
#include <QMainWindow>
#include <memory>
#include "../common/packingproblem.h"

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
    void translateCurrentToMinimumPosition();
    void createRandomLayout();
	void createBottomLeftLayout();
    void changeContainerWidth();
	void changeContainerHeight();
    void showGlobalOverlap();
    void localSearch();
    void generateCurrentTotalGlsWeightedOverlapMap();
    void updateGlsWeightedOverlapMap();
    void resetGlsWeightedOverlapMap();
    void translateCurrentToGlsWeightedMinimumPosition();
    void glsWeightedlocalSearch();

	void generateCurrentTotalSearchOverlapMap();
	void generateCurrentTotalSearchOverlapMap2();
	void generateCurrentTotalSearchOverlapMap3();
	void generateCurrentTotalSearchOverlapMap4();
    void showZoomedMap();
    void translateCurrentToMinimumZoomedPosition();
    void zoomedlocalSearch();

    void executePacking();
	void showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info);
	void showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info);
	//void showCurrent2DSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length);
	//void showCurrent2DSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, int height);
	void showExecution2DDimensionChanged(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal elapsed, uint seed);

    void saveSolution();
	void saveZoomedSolution();
    void loadSolution();
    void exportSolutionToSvg();
	void exportSolutionTikz();

	void switchToOriginalProblem();
	void updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed);

	void printDensity();

private:
    void createOverlapMessageBox(qreal globalOverlap, QVector<qreal> &individualOverlaps, qreal scale);
	int logposition(qreal value);

    Ui::MainWindow *ui;
    std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> rasterProblem, rasterZoomedProblem;
    RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver, solverGls, solverDoubleGls;
	std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> weights;
	RASTERVORONOIPACKING::RasterStripPackingParameters params;

    ZoomedMapViewDialog zoomedMapViewer;
    GlsWeightViewerDialog weightViewer;
    RunConfigurationsDialog runConfig;
	PackingThread runThread;
	Packing2DThread run2DThread;
	PackingClusterThread runClusterThread;

    int accContainerShrink;
	qreal totalArea; qreal containerWidth;
	RASTERPACKING::PackingProblem originalProblem;
};

#endif // MAINWINDOW_H
