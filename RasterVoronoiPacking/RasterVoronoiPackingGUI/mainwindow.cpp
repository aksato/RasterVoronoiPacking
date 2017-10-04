#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "packingproblem.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QTextStream>
#include <QTime>
#include <QDebug>
#include <QXmlStreamReader>
#include <QtSvg/QSvgGenerator>
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) ,
    weightViewer(this) ,
    zoomedMapViewer(this)
{
    ui->setupUi(this);

    connect(ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));
    connect(ui->actionLoad_Problem, SIGNAL(triggered()), this, SLOT(loadPuzzle()));
    connect(ui->actionLoad_Zoomed_Problem, SIGNAL(triggered()), this, SLOT(loadZoomedPuzzle()));
    connect(ui->actionLoad_Solution, SIGNAL(triggered()), this, SLOT(loadSolution()));
    connect(ui->actionSave_Solution, SIGNAL(triggered()), this, SLOT(saveSolution()));
	connect(ui->actionSave_Zoomed_Solution, SIGNAL(triggered()), this, SLOT(saveZoomedSolution()));
    connect(ui->actionExport_Solution_to_SVG, SIGNAL(triggered()), this, SLOT(exportSolutionToSvg()));
	connect(ui->actionShow_density, SIGNAL(triggered()), this, SLOT(printDensity()));
    connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), ui->graphicsView, SLOT(setSelectedItem(int)));
    connect(ui->graphicsView, SIGNAL(selectedItemChanged(int)), ui->comboBox, SLOT(setCurrentIndex(int)));
    connect(ui->spinBox, SIGNAL(valueChanged(int)), ui->graphicsView, SLOT(setSelectedItem(int)));
    connect(ui->graphicsView, SIGNAL(selectedItemChanged(int)), ui->spinBox, SLOT(setValue(int)));
    connect(ui->graphicsView, SIGNAL(selectedItemChanged(int)), this, SLOT(updateAngleComboBox(int)));
    connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateAngleComboBox(int)));
    connect(ui->graphicsView, SIGNAL(currentOrientationChanged(int)), ui->comboBox_2, SLOT(setCurrentIndex(int)));
    connect(ui->comboBox_2, SIGNAL(currentIndexChanged(int)), ui->graphicsView, SLOT(setCurrentOrientation(int)));
    connect(ui->graphicsView, SIGNAL(currentPositionChanged(QPointF)), this, SLOT(updatePositionValues(QPointF)));
    connect(ui->doubleSpinBox, SIGNAL(valueChanged(double)), ui->graphicsView, SLOT(setCurrentXCoord(double)));
    connect(ui->doubleSpinBox_2, SIGNAL(valueChanged(double)), ui->graphicsView, SLOT(setCurrentYCoord(double)));

    connect(ui->pushButton_3, SIGNAL(clicked()), this, SLOT(generateCurrentTotalOverlapMap()));
    connect(ui->pushButton_4, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumPosition()));
    connect(ui->pushButton_5, SIGNAL(clicked()), this, SLOT(createRandomLayout()));
	connect(ui->pushButton_32, SIGNAL(clicked()), this, SLOT(createBottomLeftLayout()));
    connect(ui->pushButton_14, SIGNAL(clicked()), this, SLOT(changeContainerWidth()));
	connect(ui->pushButton_19, SIGNAL(clicked()), this, SLOT(changeContainerHeight()));
    connect(ui->pushButton_6, SIGNAL(clicked()), this, SLOT(showGlobalOverlap()));
    connect(ui->pushButton_7, SIGNAL(clicked()), this, SLOT(localSearch()));
    connect(ui->pushButton_10, SIGNAL(clicked()), this, SLOT(generateCurrentTotalGlsWeightedOverlapMap()));
    connect(ui->pushButton_9, SIGNAL(clicked()), this, SLOT(translateCurrentToGlsWeightedMinimumPosition()));
    connect(ui->pushButton_8, SIGNAL(clicked()), this, SLOT(glsWeightedlocalSearch()));
    connect(ui->pushButton_12, SIGNAL(clicked()), this, SLOT(updateGlsWeightedOverlapMap()));
    connect(ui->pushButton_13, SIGNAL(clicked()), this, SLOT(resetGlsWeightedOverlapMap()));
    connect(&weightViewer, SIGNAL(weightViewerSelectionChanged(int,int)), ui->graphicsView, SLOT(highlightPair(int,int)));
    connect(ui->pushButton, SIGNAL(clicked()), &runConfig, SLOT(exec()));
    connect(&runConfig, &RunConfigurationsDialog::accepted, this, &MainWindow::executePacking);

    qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	connect(&runThread, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution,int)), this, SLOT(showCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution,int)));
    connect(&runThread, SIGNAL(weightsChanged()), &weightViewer, SLOT(updateImage()));
	connect(&runThread, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(showExecutionStatus(int, int, int, qreal, qreal, qreal)));
	connect(&runThread, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)), this, SLOT(showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)));
	connect(&runThread, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)), this, SLOT(showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)));
	connect(ui->pushButton_2, SIGNAL(clicked()), &runThread, SLOT(abort()));

	qRegisterMetaType<Solution2DInfo>("Solution2DInfo");
	connect(&run2DThread, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution, int, int)), this, SLOT(showCurrent2DSolution(RASTERVORONOIPACKING::RasterPackingSolution, int, int)));
	connect(&run2DThread, SIGNAL(weightsChanged()), &weightViewer, SLOT(updateImage()));
	connect(&run2DThread, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(showExecutionStatus(int, int, int, qreal, qreal, qreal)));
	connect(&run2DThread, SIGNAL(dimensionUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, const Solution2DInfo, int, qreal, uint)), this, SLOT(showExecution2DDimensionChanged(const RASTERVORONOIPACKING::RasterPackingSolution, const Solution2DInfo, int, qreal, uint)));
	connect(&run2DThread, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)), this, SLOT(showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)));
	connect(ui->pushButton_2, SIGNAL(clicked()), &run2DThread, SLOT(abort()));

	connect(&runClusterThread, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution, int)), this, SLOT(showCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution, int)));
	connect(&runClusterThread, SIGNAL(weightsChanged()), &weightViewer, SLOT(updateImage()));
	connect(&runClusterThread, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(showExecutionStatus(int, int, int, qreal, qreal, qreal)));
	connect(&runClusterThread, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)), this, SLOT(showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)));
	connect(&runClusterThread, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)), this, SLOT(showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)));
	connect(ui->pushButton_2, SIGNAL(clicked()), &runClusterThread, SLOT(abort()));
	connect(&runClusterThread, SIGNAL(unclustered(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)), this, SLOT(updateUnclusteredProblem(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)));

    connect(ui->pushButton_15, SIGNAL(clicked()), this, SLOT(printCurrentSolution()));
	connect(ui->pushButton_18, SIGNAL(clicked()), this, SLOT(generateCurrentTotalSearchOverlapMap()));
    connect(ui->pushButton_16, SIGNAL(clicked()), this, SLOT(showZoomedMap()));
    connect(ui->pushButton_17, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumZoomedPosition()));
    connect(ui->pushButton_20, SIGNAL(clicked()), this, SLOT(zoomedlocalSearch()));

	connect(ui->pushButton_11, SIGNAL(clicked()), this, SLOT(switchToOriginalProblem()));

    ui->comboBox->setVisible(false);
    ui->comboBox_3->setVisible(false);
    ui->graphicsView->setBackgroundBrush(QBrush(qRgb(240,240,240)));
    qsrand(4939495);
}

MainWindow::~MainWindow()
{
    delete ui;
}

qreal getContainerWidth(RASTERPACKING::PackingProblem &problem) {
	std::shared_ptr<RASTERPACKING::Polygon> conainerPolygon = (*problem.ccbegin())->getPolygon();
	qreal minY, maxY;
	minY = conainerPolygon->at(0).y(); maxY = minY;
	for (int i = 0; i < conainerPolygon->size(); i++) {
		qreal curY = conainerPolygon->at(i).y();
		if (curY < minY) minY = curY;
		if (curY > maxY) maxY = curY;
	}
	return maxY - minY;
}

int MainWindow::logposition(qreal value) {
	// set minv, ... like above
	// ...

	// position will be between 0 and 100
	qreal minp = 0;
	qreal maxp = 99;

	// The result should be between 100 an 10000000
	qreal minv = qLn(0.1);
	qreal maxv = qLn(100);

	// calculate adjustment factor
	qreal scale = (maxv - minv) / (maxp - minp);

	return (int)((qLn(value) - minv) / scale + minp);
}

void MainWindow::loadPuzzle() {
	// TOREMOVE


    QString  fileName = QFileDialog::getOpenFileName(this, tr("Open Puzzle"), "", tr("Modified ESICUP Files (*.xml)"));
    QDir::setCurrent(QFileInfo(fileName).absolutePath());
    RASTERPACKING::PackingProblem problem;
    if(problem.load(fileName)) {
		if (problem.loadClusterInfo(fileName)) {
			originalProblem.load(problem.getOriginalProblem());
			rasterProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem>(new RASTERVORONOIPACKING::RasterPackingClusterProblem);
			ui->pushButton_11->setEnabled(true);
			runConfig.enableCluster();
		}
		else rasterProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
		
		// Density calculation
		totalArea = problem.getTotalItemsArea();
		containerWidth = getContainerWidth(problem);
		
		rasterProblem->load(problem);

		solution = RASTERVORONOIPACKING::RasterPackingSolution(rasterProblem->count());

		solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D>(new RASTERVORONOIPACKING::RasterStripPackingSolver2D(rasterProblem));
		solverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(rasterProblem));

        ui->graphicsView->setEnabled(true);
        ui->graphicsView->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
        ui->graphicsView->setStatusBar(ui->statusBar);
        ui->graphicsView->createGraphicItems(problem);
		
		qreal widthFullZoom = (qreal)ui->graphicsView->width() / ui->graphicsView->sceneRect().width();
		qreal heightFullZoom = (qreal)ui->graphicsView->height() / ui->graphicsView->sceneRect().height();
		ui->graphicsView->setZoomPosition(logposition(0.7*std::min(widthFullZoom, heightFullZoom)));
        
		ui->graphicsView->setFocus();
        ui->pushButton->setEnabled(true);
        ui->pushButton_2->setEnabled(true);

        ui->spinBox->setEnabled(true); ui->spinBox->setMinimum(0); ui->spinBox->setMaximum(rasterProblem->count()-1);
        ui->comboBox->setEnabled(true);
        ui->doubleSpinBox->setEnabled(true); ui->doubleSpinBox->setSingleStep(1/ui->graphicsView->getScale());
        ui->doubleSpinBox_2->setEnabled(true); ui->doubleSpinBox_2->setSingleStep(1/ui->graphicsView->getScale());
        ui->comboBox_2->setEnabled(true);
        ui->pushButton_3->setEnabled(true); ui->pushButton_4->setEnabled(true);
        ui->pushButton_5->setEnabled(true); ui->pushButton_6->setEnabled(true);
        ui->pushButton_7->setEnabled(true); ui->pushButton_8->setEnabled(true);
		ui->pushButton_9->setEnabled(true); ui->pushButton_10->setEnabled(true); 
		ui->pushButton_12->setEnabled(true); ui->pushButton_13->setEnabled(true); 
		ui->pushButton_14->setEnabled(true); ui->pushButton_15->setEnabled(true);
		ui->pushButton_19->setEnabled(true);
		ui->pushButton_32->setEnabled(true);
		
        ui->actionLoad_Zoomed_Problem->setEnabled(true);

		ui->actionShow_density->setEnabled(true);

		weightViewer.setWeights(solverGls->getGlsWeights(), solution.getNumItems());
        runConfig.setInitialLenght((qreal)rasterProblem->getContainerWidth()/ui->graphicsView->getScale(), 1.0/ui->graphicsView->getScale());
    }
    else {
       // Display error message
    }

    accContainerShrink = 0;
}

void MainWindow::loadZoomedPuzzle() {
    QString  fileName = QFileDialog::getOpenFileName(this, tr("Open Puzzle"), "", tr("Modified ESICUP Files (*.xml)"));
    QDir::setCurrent(QFileInfo(fileName).absolutePath());
    RASTERPACKING::PackingProblem problem;
    if(problem.load(fileName)) {

		if (problem.loadClusterInfo(fileName)) rasterZoomedProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem>(new RASTERVORONOIPACKING::RasterPackingClusterProblem);
		else {
			ui->pushButton_11->setEnabled(false);
			runConfig.disableCluster();
			rasterZoomedProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
		}
        //rasterZoomedProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
        rasterZoomedProblem->load(problem);
        
		solverDoubleGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(rasterProblem, rasterZoomedProblem));
		resetGlsWeightedOverlapMap();
		solverDoubleGls->updateMapsLength(solver->getCurrentWidth(), params);

        ui->pushButton_16->setEnabled(true);
        ui->pushButton_17->setEnabled(true);
        ui->pushButton_18->setEnabled(true);
        ui->pushButton_20->setEnabled(true);
    }
    else {
       // Display error message
    }
}

void MainWindow::updateAngleComboBox(int id) {
    QVector<int> angles = ui->graphicsView->getItemAngles(id);

    int prevAngle = ui->graphicsView->getItemAngle(id);
    QPointF prevPos = ui->graphicsView->getItemPos(id);
    ui->comboBox_2->clear();
    std::for_each(angles.begin(), angles.end(), [this](int val){this->ui->comboBox_2->addItem(QString::number(val));});

    // FIXME: Needs improvement
    ui->comboBox_2->setCurrentIndex(prevAngle);
    ui->graphicsView->setItemPos(prevPos, id);
}

void MainWindow::updatePositionValues(QPointF pos) {
    ui->doubleSpinBox->setValue(pos.x());
    ui->doubleSpinBox_2->setValue(pos.y());
}

void MainWindow::printCurrentSolution() {
    ui->graphicsView->getCurrentSolution(solution);
    qDebug() << solution;
}

void MainWindow::generateCurrentTotalOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	params.setDoubleResolution(false); params.setHeuristic(NONE);
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, params);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::createRandomLayout() {
	params.setDoubleResolution(false); params.setHeuristic(NONE);
    solver->generateRandomSolution(solution, params);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::createBottomLeftLayout() {
	params.setDoubleResolution(false); params.setHeuristic(NONE);
	QTime myTimer; myTimer.start();
	solver->generateBottomLeftSolution(solution, params);
	solverGls->updateMapsLength(solver->getCurrentWidth(), params);
	if (solverDoubleGls) solverDoubleGls->updateMapsLength(solver->getCurrentWidth(), params);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth(), solver->getCurrentHeight());
	ui->graphicsView->setCurrentSolution(solution);
	ui->statusBar->showMessage("New bottom left solution created. Length: " + QString::number(solver->getCurrentWidth() / rasterProblem->getScale()) + ". Elapsed Time : " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::translateCurrentToMinimumPosition() {
    qreal minVal;
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	params.setDoubleResolution(false); params.setHeuristic(NONE);
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, params);
	QPoint minPos = solver->getMinimumOverlapPosition(curMap, minVal, BOTTOMLEFT_POS);
	int milliseconds = myTimer.elapsed();
	solution.setPosition(itemId, minPos);
    ui->graphicsView->setCurrentSolution(solution);
    ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Minimum position determined. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::createOverlapMessageBox(qreal globalOverlap, QVector<qreal> &individualOverlaps, qreal scale) {
    globalOverlap /= scale;
    QString log; QTextStream stream(&log);
    stream <<  "Starting global overlap evaluation...\n";
    stream <<  "Determining overlap for each item:\n";

    int numNonColliding = 0;
    for(int i = 0; i < rasterProblem->count(); i++) {
        qreal curOverlap = individualOverlaps[i]/scale;
        stream <<  "Item " << i << ": " << curOverlap;
        if(qFuzzyCompare(1 + 0.0, 1 + curOverlap)) {
            stream << " (no overlap)"; numNonColliding++;
        }
        stream << "\n";
    }
    stream <<  "Operation completed.\n";
    if(qFuzzyCompare(1 + 0.0, 1 + globalOverlap))
        stream <<  "Feasible layout! Zero overlap.\n";
    else {
        stream <<  "Unfeasible layout.\n";
        stream <<  "Total overlap: " << globalOverlap << ". Average overlap: " << globalOverlap/(qreal)rasterProblem->count() << ".\n";
        stream <<   numNonColliding << " non-colliding items and " << rasterProblem->count()-numNonColliding << " overlapped items.\n";
    }

    QMessageBox msgBox;
    msgBox.setText("Global overlap evaluation completed successfully!");
    msgBox.setInformativeText("The global overlap value is " + QString::number(globalOverlap));
    msgBox.setDetailedText(log);
    msgBox.exec();
}

void MainWindow::showGlobalOverlap() {
    ui->graphicsView->getCurrentSolution(solution);
    QVector<qreal> overlaps;
    qreal globalOverlap = solver->getGlobalOverlap(solution,overlaps, params);
    createOverlapMessageBox(globalOverlap, overlaps, this->rasterProblem->getScale());
}

void MainWindow::localSearch() {
    ui->graphicsView->getCurrentSolution(solution);
	params.setDoubleResolution(false); params.setHeuristic(NONE); params.setPlacementCriteria(BOTTOMLEFT_POS);
    QTime myTimer; myTimer.start();
	solver->performLocalSearch(solution, params);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::generateCurrentTotalGlsWeightedOverlapMap() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	params.setDoubleResolution(false); params.setHeuristic(GLS);
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solverGls->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, params);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	weightViewer.updateImage();
	weightViewer.show();
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::updateGlsWeightedOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
    solverGls->updateWeights(solution, params);
	if (solverDoubleGls) solverDoubleGls->updateWeights(solution, params);
    ui->statusBar->showMessage("Weights updated.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::resetGlsWeightedOverlapMap() {
    solverGls->resetWeights();
	if (solverDoubleGls) solverDoubleGls->resetWeights();
    ui->statusBar->showMessage("Weights reseted.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::translateCurrentToGlsWeightedMinimumPosition() {
	qreal minVal;
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	params.setDoubleResolution(false); params.setHeuristic(GLS);
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solverGls->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, params);
	QPoint minPos = solver->getMinimumOverlapPosition(curMap, minVal, BOTTOMLEFT_POS);
	int milliseconds = myTimer.elapsed();
	solution.setPosition(itemId, minPos);
	ui->graphicsView->setCurrentSolution(solution);
	ui->graphicsView->showTotalOverlapMap(curMap);
	weightViewer.updateImage();
	weightViewer.show();
	ui->statusBar->showMessage("Minimum position determined. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::glsWeightedlocalSearch() {
    ui->graphicsView->getCurrentSolution(solution);
	params.setDoubleResolution(false); params.setHeuristic(GLS);
    QTime myTimer; myTimer.start();
	solverGls->performLocalSearch(solution, params);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);
    weightViewer.updateImage();
    weightViewer.show();
}


void MainWindow::executePacking() {
	weightViewer.updateImage();
	weightViewer.show();

	params.setNmo(runConfig.getMaxWorse()); params.setTimeLimit(runConfig.getMaxSeconds());
	params.setFixedLength(!runConfig.getStripPacking());
	params.setDoubleResolution(runConfig.getMetaheuristic() == 2);
	params.setRectangularPacking(runConfig.getSquaredOpenDimensions());
	if (runConfig.getMetaheuristic() == 0) params.setHeuristic(NONE);
	if (runConfig.getMetaheuristic() == 1 || runConfig.getMetaheuristic() == 2) params.setHeuristic(GLS);
	if (runConfig.getInitialSolution() == 0) params.setInitialSolMethod(KEEPSOLUTION);
	if (runConfig.getInitialSolution() == 1) params.setInitialSolMethod(RANDOMFIXED);
	if (runConfig.getInitialSolution() == 2) params.setInitialSolMethod(BOTTOMLEFT);

	params.setClusterFactor(runConfig.getClusterFactor());
	if (params.isRectangularPacking() || runConfig.getMinimalRectangleProblem()) {
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> solver2D = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterStripPackingSolver2D>(solver);
		if (runConfig.getMinimalRectangleProblem()) {
			run2DThread.setParameters(params);
			run2DThread.setSolver(solver2D);
			run2DThread.setMethod(RASTERVORONOIPACKING::RANDOM_ENCLOSED);
			run2DThread.start();
		}
		else {
			run2DThread.setParameters(params);
			run2DThread.setSolver(solver2D);
			run2DThread.setMethod(RASTERVORONOIPACKING::SQUARE);
			run2DThread.start();
		}
	}
	else if (params.getClusterFactor() < 0) {
		runThread.setParameters(params);
		if (params.getHeuristic() == GLS)
		if (!params.isDoubleResolution()) runThread.setSolver(solverGls);
		else runThread.setSolver(solverDoubleGls);
		else runThread.setSolver(solver);

		// Resize container
		if (runConfig.getInitialSolution() == 0 || runConfig.getInitialSolution() == 1) {
			solver->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			solverGls->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			if (solverDoubleGls) solverDoubleGls->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			runThread.setInitialSolution(solution);
		}
		ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());

		runThread.start();
	}
	else {
		if (params.getHeuristic() != GLS) {
			qDebug() << "Non-GLS unsupported for cluster at the moment!";
			return;
		}

		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> clusterSolverGls;
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> originalSolverGls;
		if (params.isDoubleResolution()) {
			// Create new problems
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterSearchProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterZoomedProblem);
			clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS(clusterProblem, clusterSearchProblem));
			originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(clusterProblem->getOriginalProblem(), clusterSearchProblem->getOriginalProblem()));
		}
		else {
			// Create new problem
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
			clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS(clusterProblem));
			originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(clusterProblem->getOriginalProblem()));
		}

		// Configure Thread
		runClusterThread.setParameters(params);
		runClusterThread.setSolver(originalSolverGls, clusterSolverGls);

		// Resize container
		if (runConfig.getInitialSolution() == 0 || runConfig.getInitialSolution() == 1) {
			solver->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			solverGls->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			if (solverDoubleGls) solverDoubleGls->updateMapsLength(qRound(runConfig.getLenght()*ui->graphicsView->getScale()), params);
			runClusterThread.setInitialSolution(solution);
		}
		ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());

		runClusterThread.start();
	}
}

void MainWindow::changeContainerWidth() {
    // FIXME: Create custom dialog
	ui->graphicsView->getCurrentSolution(solution);
	bool ok;
    qreal lenght = QInputDialog::getDouble(this, "New container lenght", "Lenght:", (qreal)solver->getCurrentWidth()/ui->graphicsView->getScale(), 0, (qreal)10*this->rasterProblem->getContainerWidth()/ui->graphicsView->getScale(), 2, &ok);
	if (!ok) return;
    int scaledWidth = qRound(lenght*ui->graphicsView->getScale());
    solver->setContainerWidth(scaledWidth, solution, params);
	solverGls->setContainerWidth(scaledWidth, solution, params);
	if (solverDoubleGls) solverDoubleGls->setContainerWidth(scaledWidth, solution, params);
    ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::changeContainerHeight() {
	// FIXME: Create custom dialog
	ui->graphicsView->getCurrentSolution(solution);
	bool ok;
	qreal height = QInputDialog::getDouble(this, "New container height", "Height:", (qreal)solver->getCurrentHeight() / ui->graphicsView->getScale(), 0, (qreal)10 * (qreal)solver->getCurrentHeight() / ui->graphicsView->getScale(), 2, &ok);
	if (!ok) return;
	int scaledHeight = qRound(height*ui->graphicsView->getScale());
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> solver2D = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterStripPackingSolver2D>(solver);
	int currentWidth = solver->getCurrentWidth();
	solver2D->setContainerDimensions(currentWidth, scaledHeight, solution, params);
	ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth(), solver->getCurrentHeight());
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showZoomedMap() {
	int zoomSquareSize = 3 * qRound(this->rasterProblem->getScale() / this->rasterZoomedProblem->getScale());
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = solverDoubleGls->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution);
    QPixmap zoomImage = QPixmap::fromImage(zoomMap->getImage());
    zoomedMapViewer.setImage(zoomImage);
    zoomedMapViewer.show();
}

void MainWindow::translateCurrentToMinimumZoomedPosition() {
	int zoomSquareSize = 3 * qRound(this->rasterProblem->getScale() / this->rasterZoomedProblem->getScale());
	ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();

	qreal minValue; QPoint minPos; int minAngle = 0;
	minPos = solverDoubleGls->getMinimumOverlapPosition(itemId, minAngle, solution, minValue, params);
	for (uint curAngle = 1; curAngle < this->rasterProblem->getItem(itemId)->getAngleCount(); curAngle++) {
		qreal curValue; QPoint curPos;
		curPos = solverDoubleGls->getMinimumOverlapPosition(itemId, curAngle, solution, curValue, params);
		if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
	}
	solution.setOrientation(itemId, minAngle);
	solution.setPosition(itemId, minPos);
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = solverDoubleGls->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution);

    ui->graphicsView->setCurrentSolution(solution);
    QPixmap zoomImage = QPixmap::fromImage(zoomMap->getImage());
    zoomedMapViewer.setImage(zoomImage);
    zoomedMapViewer.show();
}

void MainWindow::generateCurrentTotalSearchOverlapMap() {
	ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solverDoubleGls->getTotalOverlapSearchMap(itemId, solution.getOrientation(itemId), solution, params);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap, qRound(this->rasterProblem->getScale() / this->rasterZoomedProblem->getScale()));
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::zoomedlocalSearch() {
	ui->graphicsView->getCurrentSolution(solution);
	params.setDoubleResolution(true); params.setHeuristic(GLS);
    QTime myTimer; myTimer.start();
	solverDoubleGls->performLocalSearch(solution, params);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length) {
	ui->graphicsView->recreateContainerGraphics(length);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showCurrent2DSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length) {
	ui->graphicsView->recreateContainerGraphics(length, length);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showCurrent2DSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, int height) {
	ui->graphicsView->recreateContainerGraphics(length, height);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed) {
	qreal zoomscale = params.isDoubleResolution() ? rasterZoomedProblem->getScale() : rasterProblem->getScale();
    statusBar()->showMessage("Iteration: " + QString::number(totalItNum) + " (" + QString::number(worseSolutionsCount) +
		"). Overlap: " + QString::number(curOverlap / zoomscale) + ". Minimum overlap: " + QString::number(minOverlap / zoomscale) +
		". Time elapsed: " + QString::number(elapsed) + " secs. Current Length: " + QString::number((qreal)curLength / rasterProblem->getScale()) + ").");
}

void MainWindow::showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed, uint seed) {
	qreal zoomscale = params.isDoubleResolution() ? rasterZoomedProblem->getScale() : rasterProblem->getScale();
    statusBar()->showMessage("Finished. Total iterations: " + QString::number(totalItNum) +
							". Minimum overlap: " + QString::number(minOverlap / zoomscale) +
							". Elapsed time: " + QString::number(elapsed) +
							" secs. Solution length : " + QString::number((qreal)minLength / rasterProblem->getScale()) + ".");
	this->solution = solution;
	showCurrentSolution(solution, minLength);
	solver->setContainerWidth(minLength, this->solution, params);
	solverGls->setContainerWidth(minLength, this->solution, params);
	runConfig.setInitialLenght((qreal)minLength / ui->graphicsView->getScale(), 1.0 / ui->graphicsView->getScale());
	if (solverDoubleGls) solverDoubleGls->setContainerWidth(minLength, this->solution, params);
}

void MainWindow::saveSolution() {
    QString  fileName = QFileDialog::getSaveFileName(this, tr("Save solution"), "", tr("Modified ESICUP Files (*.xml)"));
	solution.save(fileName, this->rasterProblem, solver->getCurrentWidth() / this->rasterProblem->getScale(), false);
}

void MainWindow::loadSolution() {
    QString  fileName = QFileDialog::getOpenFileName(this, tr("Load solution"), "", tr("Modified ESICUP Files (*.xml)"));

    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        qCritical() << "Error: Cannot open file"
                    << ": " << qPrintable(file.errorString());
        return;
    }

    QXmlStreamReader xml;

	// Searches for minimum length solution
    xml.setDevice(&file);
	int mostCompactSolId = 0;
	qreal minLength = -1;
	int id = 0;
	while (!xml.atEnd()) {
		xml.readNext();
		if (xml.name() == "length"  && xml.tokenType() == QXmlStreamReader::StartElement) {
			qreal curLength = xml.readElementText().toFloat();
			if (minLength < 0 || curLength < minLength) {
				minLength = curLength;
				mostCompactSolId = id;
			}
			id++;
		}
	}

	file.close();
	file.open(QFile::ReadOnly | QFile::Text);
	xml.setDevice(&file);
	int itemId = 0;
	id = -1;
    while (!xml.atEnd()) {
        xml.readNext();

		if (xml.name() == "solution" && xml.tokenType() == QXmlStreamReader::StartElement) {
			id++;
		}

		if (id == mostCompactSolId) {
			if (xml.name() == "placement" && xml.tokenType() == QXmlStreamReader::StartElement) {
				int posX = qRound(this->rasterProblem->getScale()*xml.attributes().value("x").toFloat());
				int posY = qRound(this->rasterProblem->getScale()*xml.attributes().value("y").toFloat());
				solution.setPosition(itemId, QPoint(posX, posY));
				unsigned int angleId = 0;
				for (; angleId < this->rasterProblem->getItem(itemId)->getAngleCount(); angleId++)
				if (this->rasterProblem->getItem(itemId)->getAngleValue(angleId) == xml.attributes().value("angle").toInt())
					break;
				solution.setOrientation(itemId, angleId);
				itemId++;
			}

			if (xml.name() == "length"  && xml.tokenType() == QXmlStreamReader::StartElement) {
				int scaledWidth = qRound(xml.readElementText().toFloat()*this->rasterProblem->getScale());
				params.setDoubleResolution(false); params.setHeuristic(NONE);
				solver->updateMapsLength(scaledWidth, params);
				solverGls->updateMapsLength(scaledWidth, params);
				if (solverDoubleGls) solverDoubleGls->updateMapsLength(scaledWidth, params);
				ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());
			}

			if (xml.name() == "solution" && xml.tokenType() == QXmlStreamReader::EndElement) break;
		}
    }
    file.close();

    ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::exportSolutionToSvg() {
    QString  fileName = QFileDialog::getSaveFileName(this, tr("Export solution"), "", tr("Scalable Vector Graphics (*.svg)"));

     QSvgGenerator svgGen;

     svgGen.setFileName(fileName);
     svgGen.setSize(QSize(200, 200));
     svgGen.setViewBox(QRect(0, 0, 200, 200));
     svgGen.setTitle(tr("SVG Generator Example Drawing"));
     svgGen.setDescription(tr("An SVG drawing created by the SVG Generator "));

     QPainter painter( &svgGen );
     ui->graphicsView->disableItemSelection();
     ui->graphicsView->scene()->render( &painter );
     ui->graphicsView->enableItemSelection();
}

void MainWindow::printDensity() {
	qDebug() << "Total area:" << totalArea;
	qDebug() << "Container width:" << containerWidth;
	qDebug() << "Container length:" << (qreal)solver->getCurrentWidth() / rasterProblem->getScale();
	qDebug() << qSetRealNumberPrecision(2) << "Density" << qPrintable(QString::number( (100 * rasterProblem->getScale()*totalArea) / (containerWidth*(qreal)solver->getCurrentWidth()), 'f', 2)) << "%";

	QMessageBox msgBox;
	msgBox.setText("The density of the layout is " + QString::number((100 * rasterProblem->getScale()*totalArea) / (containerWidth*(qreal)solver->getCurrentWidth()), 'f', 2) + "%.");
	msgBox.exec();
}

void MainWindow::showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal elapsed, uint seed) {
	qDebug() << "New minimum length obtained: " << minLength / rasterProblem->getScale() << ". It = " << totalItNum << ". Elapsed time: " << elapsed << " secs";
}

void MainWindow::showExecution2DDimensionChanged(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const Solution2DInfo &info, int totalItNum, qreal elapsed, uint seed) {
	qDebug() << "New dimensions: " << info.length / rasterProblem->getScale() << info.height / rasterProblem->getScale() << "Area =" << (info.length * info.height) / (rasterProblem->getScale() * rasterProblem->getScale()) << ". It = " << totalItNum << ". Elapsed time: " << elapsed << " secs";
}

void MainWindow::saveZoomedSolution() {
	QString  fileName = QFileDialog::getSaveFileName(this, tr("Save solution"), "", tr("Modified ESICUP Files (*.xml)"));
	solution.save(fileName, this->rasterZoomedProblem, solver->getCurrentWidth() / this->rasterProblem->getScale(), false);
}

void MainWindow::switchToOriginalProblem() {
	// Create new problem. TODO: Support for double resolution
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
	int oldWidth = solver->getCurrentWidth();
	rasterProblem = clusterProblem->getOriginalProblem();
	if (solverDoubleGls) rasterZoomedProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterZoomedProblem)->getOriginalProblem();
	// Recreate solution
	ui->graphicsView->getCurrentSolution(solution);
	clusterProblem->convertSolution(solution);

	// Recreate items and container graphics
	solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(rasterProblem));
	solverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(rasterProblem));
	if (solverDoubleGls) solverDoubleGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(rasterProblem, rasterZoomedProblem));
	ui->graphicsView->createGraphicItems(originalProblem); ui->graphicsView->scale(1.0, -1.0);

	// Change width
	solver->setContainerWidth(oldWidth, solution, params);
	solverGls->setContainerWidth(oldWidth, solution, params);
	if (solverDoubleGls) solverDoubleGls->setContainerWidth(oldWidth, solution, params);
	ui->spinBox->setMaximum(rasterProblem->count() - 1);
	ui->graphicsView->recreateContainerGraphics(oldWidth);

	// Update ui
	ui->graphicsView->setCurrentSolution(solution);
	ui->pushButton_11->setEnabled(false);
	ui->statusBar->showMessage("Switched to original problem (cannot undo!).");
}

void MainWindow::updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed) {
	// Create new problem. TODO: Support for double resolution
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
	rasterProblem = clusterProblem->getOriginalProblem();
	if (solverDoubleGls) rasterZoomedProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterZoomedProblem)->getOriginalProblem();
	ui->graphicsView->createGraphicItems(originalProblem); ui->graphicsView->scale(1.0, -1.0);
	ui->graphicsView->recreateContainerGraphics(length);
	ui->graphicsView->setCurrentSolution(solution);
	ui->spinBox->setMaximum(rasterProblem->count() - 1);

	// Update solvers
	solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(rasterProblem));
	solverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(rasterProblem));
	if (solverDoubleGls) solverDoubleGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(rasterProblem, rasterZoomedProblem));

	// Disable cluster execution
	runConfig.disableCluster();

	// Print to console
	qDebug() << "Undoing the initial clusters, returning to original problem. Current length:" << length / rasterProblem->getScale() << ". Elapsed time: " << elapsed << " secs";
}