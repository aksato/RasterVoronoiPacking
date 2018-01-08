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
    connect(ui->actionLoad_Solution, SIGNAL(triggered()), this, SLOT(loadSolution()));
    connect(ui->actionSave_Solution, SIGNAL(triggered()), this, SLOT(saveSolution()));
    connect(ui->actionExport_Solution_to_SVG, SIGNAL(triggered()), this, SLOT(exportSolutionToSvg()));
	connect(ui->actionExport_Solution_to_Tikz, SIGNAL(triggered()), this, SLOT(exportSolutionTikz()));
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

	qRegisterMetaType<ExecutionSolutionInfo>("ExecutionSolutionInfo");
    qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");

    connect(ui->pushButton_15, SIGNAL(clicked()), this, SLOT(printCurrentSolution()));
	connect(ui->pushButton_18, SIGNAL(clicked()), this, SLOT(generateCurrentTotalSearchOverlapMap()));
	connect(ui->pushButton_22, SIGNAL(clicked()), this, SLOT(setExplicityZoomValue()));
    connect(ui->pushButton_16, SIGNAL(clicked()), this, SLOT(showZoomedMap()));
    connect(ui->pushButton_17, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumZoomedPosition()));
    connect(ui->pushButton_20, SIGNAL(clicked()), this, SLOT(zoomedlocalSearch()));

	connect(ui->pushButton_11, SIGNAL(clicked()), this, SLOT(switchToOriginalProblem()));

    ui->comboBox->setVisible(false);
    ui->comboBox_3->setVisible(false);
    ui->graphicsView->setBackgroundBrush(QBrush(qRgb(240,240,240)));
	zoomFactor = 1;
    qsrand(4939495);
}

MainWindow::~MainWindow()
{
    delete ui;
}

std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> MainWindow::createBasicSolver() {
	RasterStripPackingParameters tempParameters(RASTERVORONOIPACKING::NONE, 1);
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver({ rasterProblem },
		tempParameters, currentContainerWidth, currentContainerHeight);
	solver->overlapEvaluator = this->overlapEvaluator;
	solver->setContainerDimensions(currentContainerWidth, currentContainerHeight);
	return solver;
}

std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> MainWindow::createGLSSolver() {
	RasterStripPackingParameters tempParameters(RASTERVORONOIPACKING::GLS, 1);
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver({ rasterProblem },
		tempParameters, currentContainerWidth, currentContainerHeight);
	solverGls->overlapEvaluator = this->overlapEvaluatorGls;
	solverGls->setContainerDimensions(currentContainerWidth, currentContainerHeight);
	return solverGls;
}

std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> MainWindow::createDoubleGLSSolver() {
	RasterStripPackingParameters tempParameters(RASTERVORONOIPACKING::GLS, this->zoomFactor);
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverDoubleGls = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver({ rasterProblem },
		tempParameters, currentContainerWidth, currentContainerHeight);
	solverDoubleGls->overlapEvaluator = this->overlapEvaluatorDoubleGls;
	solverDoubleGls->setContainerDimensions(currentContainerWidth, currentContainerHeight);
	return solverDoubleGls;
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
		rasterProblem->load(problem);
		currentContainerWidth = rasterProblem->getContainerWidth(); currentContainerHeight = rasterProblem->getContainerHeight();
		solution = RASTERVORONOIPACKING::RasterPackingSolution(rasterProblem->count());
		weights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(rasterProblem->count()));
		this->overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(this->rasterProblem, std::shared_ptr<GlsNoWeightSet>(new GlsNoWeightSet), true));
		this->overlapEvaluatorGls = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(this->rasterProblem, weights, true));

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
		ui->pushButton_22->setEnabled(true);

        ui->actionLoad_Zoomed_Problem->setEnabled(true);

		ui->actionShow_density->setEnabled(true);

		weightViewer.setWeights(weights, solution.getNumItems());
        runConfig.setInitialLenght((qreal)rasterProblem->getContainerWidth()/ui->graphicsView->getScale(), 1.0/ui->graphicsView->getScale());
		runConfig.setInitialZoomFactor(rasterProblem->getScale());
    }
    else {
       // Display error message
    }

    accContainerShrink = 0;
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
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->overlapEvaluator->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution);
	
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::createRandomLayout() {
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
    solver->generateRandomSolution(solution);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::createBottomLeftLayout() {
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = zoomFactor == 1 ? createBasicSolver() : createDoubleGLSSolver();
	solver->generateBottomLeftSolution(solution);
	currentContainerWidth = solver->getCurrentWidth(); currentContainerHeight = solver->getCurrentHeight();
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->recreateContainerGraphics(currentContainerWidth);
	ui->graphicsView->setCurrentSolution(solution);
	ui->statusBar->showMessage("New bottom left solution created. Length: " + QString::number(solver->getCurrentWidth() / rasterProblem->getScale()) + ". Elapsed Time : " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::translateCurrentToMinimumPosition() {
	quint32 minVal;
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->overlapEvaluator->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution);
	QPoint minPos = solver->overlapEvaluator->getMinimumOverlapPosition(itemId, solution.getOrientation(itemId), solution, minVal);
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
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
	qreal globalOverlap = solver->getGlobalOverlap(solution);
	for (int itemId = 0; itemId < rasterProblem->count(); itemId++) {
		qreal itemOverlap = solver->getItemTotalOverlap(itemId, solution);
		overlaps.append(itemOverlap);
	}
    createOverlapMessageBox(globalOverlap, overlaps, this->rasterProblem->getScale());
}

void MainWindow::localSearch() {
    ui->graphicsView->getCurrentSolution(solution);
    QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
	solver->performLocalSearch(solution);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::generateCurrentTotalGlsWeightedOverlapMap() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = createGLSSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solverGls->overlapEvaluator->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	weightViewer.updateImage();
	weightViewer.show();
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::updateGlsWeightedOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = createGLSSolver();
	std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterTotalOverlapMapEvaluatorGLS>(solverGls->overlapEvaluator)->setgetGlsWeights(weights);
	QVector<quint32> currentOverlaps(rasterProblem->count()*rasterProblem->count()); quint32 maxItemOverlap;
	solverGls->getGlobalOverlap(solution, currentOverlaps, maxItemOverlap);
	solverGls->updateWeights(solution, currentOverlaps, maxItemOverlap);
    ui->statusBar->showMessage("Weights updated.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::resetGlsWeightedOverlapMap() {
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = createGLSSolver();
    solverGls->resetWeights();
    ui->statusBar->showMessage("Weights reseted.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::translateCurrentToGlsWeightedMinimumPosition() {
	quint32 minVal;
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = createGLSSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solverGls->overlapEvaluator->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution);
	QPoint minPos = solverGls->overlapEvaluator->getMinimumOverlapPosition(itemId, solution.getOrientation(itemId), solution, minVal);
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
    QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverGls = createGLSSolver();
	solverGls->performLocalSearch(solution);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::changeContainerWidth() {
    // FIXME: Create custom dialog
	ui->graphicsView->getCurrentSolution(solution);
	bool ok;
	qreal lenght = QInputDialog::getDouble(this, "New container lenght", "Lenght:", (qreal)currentContainerWidth/ ui->graphicsView->getScale(), 0, (qreal)10 * this->rasterProblem->getContainerWidth() / ui->graphicsView->getScale(), 2, &ok);
	if (!ok) return;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
    int scaledWidth = qRound(lenght*ui->graphicsView->getScale());
	if (scaledWidth < solver->getMinimumContainerWidth()) {
		ui->statusBar->showMessage("Could not reduce container width.");
		return;
	}
	solver->setContainerWidth(scaledWidth, solution);
	currentContainerWidth = solver->getCurrentWidth();
	ui->graphicsView->recreateContainerGraphics(currentContainerWidth);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::changeContainerHeight() {
	// FIXME: Create custom dialog
	ui->graphicsView->getCurrentSolution(solution);
	bool ok;
	qreal height = QInputDialog::getDouble(this, "New container height", "Height:", (qreal)currentContainerHeight / ui->graphicsView->getScale(), 0, (qreal)10 * (qreal)currentContainerHeight / ui->graphicsView->getScale(), 2, &ok);
	if (!ok) return;
	int scaledHeight = qRound(height*ui->graphicsView->getScale());
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = createBasicSolver();
	if (scaledHeight < solver->getMinimumContainerHeight()) {
		ui->statusBar->showMessage("Could not reduce container height.");
		return;
	}
	solver->setContainerDimensions(currentContainerWidth, scaledHeight, solution);
	currentContainerHeight = solver->getCurrentHeight();
	ui->graphicsView->recreateContainerGraphics(currentContainerWidth, currentContainerHeight);
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showZoomedMap() {
	int zoomSquareSize = ZOOMNEIGHBORHOOD * zoomFactor;
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverDoubleGls = createDoubleGLSSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterTotalOverlapMapEvaluatorDoubleGLS>(solverDoubleGls->overlapEvaluator)->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution);
    QPoint curItemPosition = solution.getPosition(itemId);
    zoomedMapViewer.getMapView()->updateMap(zoomMap, curItemPosition);
    zoomedMapViewer.show();
}

void MainWindow::translateCurrentToMinimumZoomedPosition() {
	int zoomSquareSize = ZOOMNEIGHBORHOOD * zoomFactor;
	ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();

	quint32 minValue; QPoint minPos; int minAngle = 0;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverDoubleGls = createDoubleGLSSolver();
	minPos = solverDoubleGls->overlapEvaluator->getMinimumOverlapPosition(itemId, minAngle, solution, minValue);
	for (uint curAngle = 1; curAngle < this->rasterProblem->getItem(itemId)->getAngleCount(); curAngle++) {
		quint32 curValue; QPoint curPos;
		curPos = solverDoubleGls->overlapEvaluator->getMinimumOverlapPosition(itemId, curAngle, solution, curValue);
		if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
	}
	solution.setOrientation(itemId, minAngle);
	solution.setPosition(itemId, minPos);
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterTotalOverlapMapEvaluatorDoubleGLS>(solverDoubleGls->overlapEvaluator)->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution);
    ui->graphicsView->setCurrentSolution(solution);
    QPixmap zoomImage = QPixmap::fromImage(zoomMap->getImage());
    QPoint curItemPosition = solution.getPosition(itemId);
    zoomedMapViewer.getMapView()->updateMap(zoomMap, curItemPosition);
    zoomedMapViewer.show();
}

void MainWindow::generateCurrentTotalSearchOverlapMap() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverDoubleGls = createDoubleGLSSolver();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterTotalOverlapMapEvaluatorDoubleGLS>(solverDoubleGls->overlapEvaluator)->getTotalOverlapSearchMap(itemId, solution.getOrientation(itemId), solution);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap, zoomFactor);
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::setExplicityZoomValue() {
	bool ok;
	zoomFactor = QInputDialog::getInt(this, "Explicity Zoom Value", "Zoom:", this->rasterProblem->getScale(), 1, 1000, 1, &ok);
	if (!ok) return;
	int zoomSquareSize = ZOOMNEIGHBORHOOD * zoomFactor;
	zoomedMapViewer.getMapView()->init(zoomSquareSize + 1 - zoomSquareSize % 2, 1.0 / (qreal)zoomFactor);
	ui->pushButton_16->setEnabled(true); ui->pushButton_17->setEnabled(true); ui->pushButton_18->setEnabled(true); ui->pushButton_20->setEnabled(true);
	this->overlapEvaluatorDoubleGls = std::shared_ptr<RasterTotalOverlapMapEvaluatorDoubleGLS>(new RasterTotalOverlapMapEvaluatorDoubleGLS(this->rasterProblem, zoomFactor, weights, true));
}

void MainWindow::zoomedlocalSearch() {
	ui->graphicsView->getCurrentSolution(solution);
    QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solverDoubleGls = createDoubleGLSSolver();
	solverDoubleGls->performLocalSearch(solution);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::showCurrentSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	if (info.pType == ExecutionSolutionInfo::ProblemType::StripPacking) ui->graphicsView->recreateContainerGraphics(info.length);
	else  ui->graphicsView->recreateContainerGraphics(info.length, info.height); // ExecutionSolutionInfo::ProblemType::SquarePacking || ExecutionSolutionInfo::ProblemType::RectangularPacking
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::showExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed) {
	qreal zoomscale = rasterProblem->getScale();
    statusBar()->showMessage("Iteration: " + QString::number(totalItNum) + " (" + QString::number(worseSolutionsCount) +
		"). Overlap: " + QString::number(curOverlap / zoomscale) + ". Minimum overlap: " + QString::number(minOverlap / zoomscale) +
		". Time elapsed: " + QString::number(elapsed) + " secs. Current Length: " + QString::number((qreal)curLength / rasterProblem->getScale()) + ").");
}

void MainWindow::showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed) {
	qreal zoomscale = rasterProblem->getScale();
	this->solution = solution;
	int minLength = info.length;
	showCurrentSolution(solution, info);
	runConfig.setInitialLenght((qreal)minLength / ui->graphicsView->getScale(), 1.0 / ui->graphicsView->getScale());
	if (info.pType == ExecutionSolutionInfo::ProblemType::StripPacking) {
		statusBar()->showMessage("Finished. Total iterations: " + QString::number(totalItNum) +
			". Minimum overlap: " + QString::number(minOverlap / zoomscale) +
			". Elapsed time: " + QString::number(elapsed) +
			" secs. Solution length : " + QString::number((qreal)info.length / rasterProblem->getScale()) + ".");
		currentContainerWidth = minLength;
	}
	else { // ExecutionSolutionInfo::ProblemType::SquarePacking || ExecutionSolutionInfo::ProblemType::RectangularPacking
		statusBar()->showMessage("Finished. Total iterations: " + QString::number(totalItNum) +
			". Minimum overlap: " + QString::number(minOverlap / zoomscale) +
			". Elapsed time: " + QString::number(elapsed) +
			" secs. Solution area : " + QString::number((qreal)info.area / (rasterProblem->getScale()*rasterProblem->getScale())) + ".");
		int minHeight = info.height;
		currentContainerWidth = minLength; currentContainerHeight = minHeight;
	}
}

void MainWindow::saveSolution() {
    QString  fileName = QFileDialog::getSaveFileName(this, tr("Save solution"), "", tr("Modified ESICUP Files (*.xml)"));
	solution.save(fileName, this->rasterProblem, currentContainerWidth / this->rasterProblem->getScale(), false);
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

	// Searches for minimum area solution
    xml.setDevice(&file);
	int mostCompactSolId = 0;
	int mostCompactSolAreaId = -1;
	qreal minArea = -1;
	qreal minLength = -1;
	qreal curLength = -1, curArea = -1;
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
		if (xml.name() == "area"  && xml.tokenType() == QXmlStreamReader::StartElement) {
			qreal curArea = xml.readElementText().toFloat();
			if (minArea < 0 || curArea < minArea) {
				minArea = curArea;
				mostCompactSolAreaId = id - 1;
			}
		}
	}
	if (mostCompactSolAreaId > 0) mostCompactSolId = mostCompactSolAreaId;

	file.close();
	file.open(QFile::ReadOnly | QFile::Text);
	xml.setDevice(&file);
	int itemId = 0;
	id = -1;
	bool heightdefined = false;
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
				currentContainerWidth = scaledWidth;
				
			}

			if (xml.name() == "height"  && xml.tokenType() == QXmlStreamReader::StartElement) {
				int scaledHeight = qRound(xml.readElementText().toFloat()*this->rasterProblem->getScale());
				currentContainerHeight = scaledHeight;
				heightdefined = true;
				
			}

			if (xml.name() == "solution" && xml.tokenType() == QXmlStreamReader::EndElement) {
				if(!heightdefined) ui->graphicsView->recreateContainerGraphics(currentContainerWidth);
				else ui->graphicsView->recreateContainerGraphics(currentContainerWidth, currentContainerHeight);
				break;
			}
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

void MainWindow::exportSolutionTikz() {
	QString  fileName = QFileDialog::getSaveFileName(this, tr("Export solution"), "", tr("Portable Graphics Format (*.pgf)"));
	solution.exportToPgf(fileName, this->rasterProblem, (qreal)currentContainerWidth / this->rasterProblem->getScale(), (qreal)currentContainerHeight / this->rasterProblem->getScale());
}

void MainWindow::printDensity() {
	qDebug() << "Total area:" << totalArea;
	ui->graphicsView->getCurrentSolution(solution);
	qDebug() << "Container Width:" << rasterProblem->getCurrentHeight(solution) << "(original:" << rasterProblem->getOriginalHeight() << ")";
	qDebug() << "Container Length:" << rasterProblem->getCurrentWidth(solution); // << "(original:" << rasterProblem->getOriginalWidth() << ").";
	QString densityStr = QString::number((100 * rasterProblem->getDensity(solution)), 'f', 2);
	QString density2dStr = QString::number((100 * rasterProblem->getRectangularDensity(solution)), 'f', 2);
	qDebug() << qSetRealNumberPrecision(2) << "Density:" << qPrintable(densityStr) << "%";
	qDebug() << qSetRealNumberPrecision(2) << "Density 2D:" << qPrintable(density2dStr) << "%";

	QMessageBox msgBox;
	msgBox.setText("The strip density of the layout is " + densityStr + "% and the rectangular density is " + density2dStr + "%.");
	msgBox.exec();
}

void MainWindow::showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	if (info.pType == ExecutionSolutionInfo::ProblemType::StripPacking) qDebug() << "New minimum length obtained: " << info.length / rasterProblem->getScale() << ". It = " << info.iteration << ". Elapsed time: " << info.timestamp << " secs";
	else qDebug() << "New layout obtained: " << info.length / rasterProblem->getScale() << "x" << info.height / rasterProblem->getScale() << " ( area = " << (info.length * info.height) / (rasterProblem->getScale() * rasterProblem->getScale()) << "). It = " << info.iteration << ". Elapsed time: " << info.timestamp << " secs"; // ExecutionSolutionInfo::ProblemType::SquarePacking || ExecutionSolutionInfo::ProblemType::RectangularPacking
}

void MainWindow::showExecution2DDimensionChanged(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info, int totalItNum, qreal elapsed, uint seed) {
	qDebug() << "New dimensions: " << info.length / rasterProblem->getScale() << info.height / rasterProblem->getScale() << "Area =" << (info.length * info.height) / (rasterProblem->getScale() * rasterProblem->getScale()) << ". It = " << totalItNum << ". Elapsed time: " << elapsed << " secs";
}

void MainWindow::switchToOriginalProblem() {
	// Create new problem
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
	rasterProblem = clusterProblem->getOriginalProblem();
	// Recreate solution
	ui->graphicsView->getCurrentSolution(solution);
	clusterProblem->convertSolution(solution);
	// Recreate items and container graphics
	ui->graphicsView->createGraphicItems(originalProblem); ui->graphicsView->scale(1.0, -1.0);
	ui->graphicsView->recreateContainerGraphics(currentContainerWidth);
	ui->spinBox->setMaximum(rasterProblem->count() - 1);
	// Update ui
	ui->graphicsView->setCurrentSolution(solution);
	ui->pushButton_11->setEnabled(false);
	ui->statusBar->showMessage("Switched to original problem (cannot undo!).");
}

void MainWindow::updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed) {
	// Create new problem. TODO: Support for double resolution
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(rasterProblem);
	rasterProblem = clusterProblem->getOriginalProblem();
	ui->graphicsView->createGraphicItems(originalProblem); ui->graphicsView->scale(1.0, -1.0);
	ui->graphicsView->recreateContainerGraphics(length);
	ui->graphicsView->setCurrentSolution(solution);
	ui->spinBox->setMaximum(rasterProblem->count() - 1);

	// Disable cluster execution
	runConfig.disableCluster();

	// Print to console
	qDebug() << "Undoing the initial clusters, returning to original problem. Current length:" << length / rasterProblem->getScale() << ". Elapsed time: " << elapsed << " secs";
}

void MainWindow::executePacking() {
	weightViewer.updateImage();
	weightViewer.show();

	params.setNmo(runConfig.getMaxWorse()); params.setTimeLimit(runConfig.getMaxSeconds());
	switch (runConfig.getPackingProblemIndex()) {
		case 0: params.setFixedLength(false); params.setRectangularPacking(false); break;
		case 1: params.setFixedLength(false); params.setRectangularPacking(true); params.setRectangularPackingMethod(RASTERVORONOIPACKING::SQUARE); break;
		case 2: params.setFixedLength(false); params.setRectangularPacking(true); 
			switch (runConfig.getRectangularMethod()) {
				case 0: params.setRectangularPackingMethod(RASTERVORONOIPACKING::RANDOM_ENCLOSED); break;
				case 1: params.setRectangularPackingMethod(RASTERVORONOIPACKING::COST_EVALUATION); break;
				case 2: params.setRectangularPackingMethod(RASTERVORONOIPACKING::BAGPIPE); break;
			}
			break;
		case 3: params.setFixedLength(true); params.setRectangularPacking(false); break;
	}
	if (runConfig.getMetaheuristic() == 0) params.setHeuristic(NONE);
	if (runConfig.getMetaheuristic() == 1 || runConfig.getMetaheuristic() == 2) params.setHeuristic(GLS);
	if (runConfig.isZoomedApproach()) params.setZoomFactor (runConfig.getZoomRatio());
	switch (runConfig.getInitialSolution()) {
		case 0: params.setInitialSolMethod(KEEPSOLUTION); break;
		case 1: params.setInitialSolMethod(RANDOMFIXED); break;
		case 2: params.setInitialSolMethod(BOTTOMLEFT); break;
	}
	if (runConfig.getInitialSolution() != 2) params.setInitialLenght(runConfig.getLenght());
	params.setClusterFactor(runConfig.getClusterFactor());

	// Determine initial width
	int initialWidth;
	switch (params.getInitialSolMethod()) {
		case RANDOMFIXED: initialWidth = qRound(params.getInitialLenght()*rasterProblem->getScale()); currentContainerWidth = initialWidth;  break;
		case BOTTOMLEFT: initialWidth = -1; break;
		case KEEPSOLUTION: initialWidth = currentContainerWidth;
	}

	// Determine execution type
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver(rasterProblem, params, initialWidth);
	std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterTotalOverlapMapEvaluatorGLS>(solver->overlapEvaluator)->setgetGlsWeights(weights);
	solver->resetWeights();
	bool clusterExecution = params.getClusterFactor() > 0;
	if (!clusterExecution) {
		if (params.isRectangularPacking()) packer = std::shared_ptr<Packing2DThread>(new Packing2DThread);
		else packer = std::shared_ptr<PackingThread>(new PackingThread);
	}
	else {
		packer = std::shared_ptr<PackingClusterThread>(new PackingClusterThread);
		std::shared_ptr<PackingClusterThread> packerCluster = std::dynamic_pointer_cast<PackingClusterThread>(packer);
		connect(&*packerCluster, SIGNAL(unclustered(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)), this, SLOT(updateUnclusteredProblem(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)));
	}

	// Set initial solution
	if (params.getInitialSolMethod() == KEEPSOLUTION) packer->setInitialSolution(solution);

	// Configure packer object
	connect(&*packer, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)), this, SLOT(showCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)));
	connect(&*packer, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)), this, SLOT(showCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)));
	connect(&*packer, SIGNAL(weightsChanged()), &weightViewer, SLOT(updateImage()));
	connect(&*packer, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(showExecutionStatus(int, int, int, qreal, qreal, qreal)));
	connect(&*packer, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)), this, SLOT(showExecutionMinLengthObtained(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)));
	connect(&*packer, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)), this, SLOT(showExecutionFinishedStatus(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)));
	connect(ui->pushButton_2, SIGNAL(clicked()), &*packer, SLOT(abort()));
	packer->setSolver(solver);
	packer->setParameters(params);
	packer->start();
}
