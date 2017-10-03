#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rasterstrippackingsolvergls.h"
#include "raster/rasterstrippackingsolverdoublegls.h"
#include "raster/packingthread.h"
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
    void showGlobalOverlap();
    void localSearch();
    void generateCurrentTotalGlsWeightedOverlapMap();
    void updateGlsWeightedOverlapMap();
    void resetGlsWeightedOverlapMap();
    void translateCurrentToGlsWeightedMinimumPosition();
    void glsWeightedlocalSearch();

	void generateCurrentTotalSearchOverlapMap();
    void showZoomedMap();
    void translateCurrentToMinimumZoomedPosition();
    void zoomedlocalSearch();

    void executePacking();
	void showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length);
	void showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed, uint seed);
	void showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal elapsed, uint seed);

    void saveSolution();
	void saveZoomedSolution();
    void loadSolution();
    void exportSolutionToSvg();

	void switchToOriginalProblem();
	void updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed);

	void printDensity();

private:
    void createOverlapMessageBox(qreal globalOverlap, QVector<qreal> &individualOverlaps, qreal scale);
	int logposition(qreal value);

    Ui::MainWindow *ui;
    std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> rasterProblem, rasterZoomedProblem;
    RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS> solverGls;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS> solverDoubleGls;
	RASTERVORONOIPACKING::RasterStripPackingParameters params;

    ZoomedMapViewDialog zoomedMapViewer;
    GlsWeightViewerDialog weightViewer;
    RunConfigurationsDialog runConfig;
	PackingThread runThread;
	PackingClusterThread runClusterThread;

    int accContainerShrink;
	qreal totalArea; qreal containerWidth;
	RASTERPACKING::PackingProblem originalProblem;
};

#endif // MAINWINDOW_H
