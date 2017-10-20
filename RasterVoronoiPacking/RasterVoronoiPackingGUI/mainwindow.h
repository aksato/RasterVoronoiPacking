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
	void setExplicityZoomValue();
    void showZoomedMap();
    void translateCurrentToMinimumZoomedPosition();
    void zoomedlocalSearch();

    void executePacking();
	void showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info);
	void showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info);
	void showExecution2DDimensionChanged(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal elapsed, uint seed);

    void saveSolution();
    void loadSolution();
    void exportSolutionToSvg();
	void exportSolutionTikz();

	void switchToOriginalProblem();
	void updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed);

	void printDensity();

private:
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> createBasicSolver();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> createGLSSolver();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> createDoubleGLSSolver();
    void createOverlapMessageBox(qreal globalOverlap, QVector<qreal> &individualOverlaps, qreal scale);
	int logposition(qreal value);

    Ui::MainWindow *ui;
    std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> rasterProblem;
    RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::GlsWeightSet> weights;
	RASTERVORONOIPACKING::RasterStripPackingParameters params;

    ZoomedMapViewDialog zoomedMapViewer;
    GlsWeightViewerDialog weightViewer;
    RunConfigurationsDialog runConfig;

	int accContainerShrink;
	qreal searchScale;
	qreal totalArea;
	RASTERPACKING::PackingProblem originalProblem;
	int currentContainerWidth, currentContainerHeight;
	std::shared_ptr<PackingThread> packer;
};

#endif // MAINWINDOW_H
