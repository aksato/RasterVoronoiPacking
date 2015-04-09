#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "packingproblem.h"
#include "../common/cuda/gpuinfo.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QTextStream>
#include <QTime>
#include <QDebug>
#include <QXmlStreamReader>
#include <QtSvg/QSvgGenerator>

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
    connect(ui->actionExport_Solution_to_SVG, SIGNAL(triggered()), this, SLOT(exportSolutionToSvg()));
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
	connect(ui->pushButton_11, SIGNAL(clicked()), this, SLOT(generateCurrentTotalOverlapMapGPU()));
	connect(ui->pushButton_29, SIGNAL(clicked()), this, SLOT(generateCurrentTotalOverlapMapCache()));
    connect(ui->pushButton_4, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumPosition()));
	connect(ui->pushButton_21, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumPositionGPU()));
    connect(ui->pushButton_5, SIGNAL(clicked()), this, SLOT(createRandomLayout()));
    connect(ui->pushButton_14, SIGNAL(clicked()), this, SLOT(changeContainerWidth()));
    connect(ui->pushButton_6, SIGNAL(clicked()), this, SLOT(showGlobalOverlap()));
    connect(ui->pushButton_7, SIGNAL(clicked()), this, SLOT(localSearch()));
	connect(ui->pushButton_22, SIGNAL(clicked()), this, SLOT(localSearchGPU()));
	connect(ui->pushButton_26, SIGNAL(clicked()), this, SLOT(localSearchCache()));
	connect(ui->pushButton_28, SIGNAL(clicked()), this, SLOT(printCacheInfo()));
	connect(ui->pushButton_30, SIGNAL(clicked()), this, SLOT(printGlsWeightedCacheInfo()));
    connect(ui->pushButton_10, SIGNAL(clicked()), this, SLOT(generateCurrentTotalGlsWeightedOverlapMap()));
	connect(ui->pushButton_23, SIGNAL(clicked()), this, SLOT(generateCurrentTotalGlsWeightedOverlapMapGPU()));
	connect(ui->pushButton_31, SIGNAL(clicked()), this, SLOT(generateCurrentTotalGlsWeightedOverlapMapCache()));
    connect(ui->pushButton_9, SIGNAL(clicked()), this, SLOT(translateCurrentToGlsWeightedMinimumPosition()));
	connect(ui->pushButton_24, SIGNAL(clicked()), this, SLOT(translateCurrentToGlsWeightedMinimumPositionGPU()));
    connect(ui->pushButton_8, SIGNAL(clicked()), this, SLOT(glsWeightedlocalSearch()));
	connect(ui->pushButton_25, SIGNAL(clicked()), this, SLOT(glsWeightedlocalSearchGPU()));
	connect(ui->pushButton_27, SIGNAL(clicked()), this, SLOT(glsWeightedlocalSearchCache()));
    connect(ui->pushButton_12, SIGNAL(clicked()), this, SLOT(updateGlsWeightedOverlapMap()));
    connect(ui->pushButton_13, SIGNAL(clicked()), this, SLOT(resetGlsWeightedOverlapMap()));
    connect(&weightViewer, SIGNAL(weightViewerSelectionChanged(int,int)), ui->graphicsView, SLOT(highlightPair(int,int)));
    connect(ui->pushButton, SIGNAL(clicked()), &runConfig, SLOT(exec()));
    connect(&runConfig, &RunConfigurationsDialog::accepted, this, &MainWindow::executePacking);

    qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
    connect(&runThread, SIGNAL(solutionGenerated(RASTERVORONOIPACKING::RasterPackingSolution,qreal)), ui->graphicsView, SLOT(setCurrentSolution(RASTERVORONOIPACKING::RasterPackingSolution,qreal)));
    connect(&runThread, SIGNAL(weightsChanged()), &weightViewer, SLOT(updateImage()));
    connect(&runThread, SIGNAL(statusUpdated(int,int,qreal,qreal,qreal,qreal)), this, SLOT(showExecutionStatus(int,int,qreal,qreal,qreal,qreal)));
    connect(&runThread, SIGNAL(finishedExecution(int,qreal,qreal,qreal,qreal)), this, SLOT(showExecutionFinishedStatus(int,qreal,qreal,qreal,qreal)));
    connect(ui->pushButton_2, SIGNAL(clicked()), &runThread, SLOT(terminate())); // FIXME: Using terminate() is not recommended

    connect(ui->pushButton_15, SIGNAL(clicked()), this, SLOT(printCurrentSolution()));
    connect(ui->pushButton_16, SIGNAL(clicked()), this, SLOT(showZoomedMap()));
    connect(ui->pushButton_17, SIGNAL(clicked()), this, SLOT(translateCurrentToMinimumZoomedPosition()));
    connect(ui->pushButton_18, SIGNAL(clicked()), this, SLOT(showZoomedGlobalOverlap()));
    connect(ui->pushButton_19, SIGNAL(clicked()), this, SLOT(updateZoomedGlsWeights()));
    connect(ui->pushButton_20, SIGNAL(clicked()), this, SLOT(zoomedlocalSearch()));

    ui->comboBox->setVisible(false);
    ui->comboBox_3->setVisible(false);
    ui->graphicsView->setBackgroundBrush(QBrush(qRgb(240,240,240)));
    qsrand(4939495);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::loadPuzzle() {
	// TOREMOVE


    QString  fileName = QFileDialog::getOpenFileName(this, tr("Open Puzzle"), "", tr("Modified ESICUP Files (*.xml)"));
    QDir::setCurrent(QFileInfo(fileName).absolutePath());
    RASTERPREPROCESSING::PackingProblem problem;
    if(problem.load(fileName)) {
        rasterProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);

		// Get GPU memory requirements
		bool loadGPU = false;
		int numGPUs;  size_t freeCUDAMem, totalCUDAmem;
		if (CUDAPACKING::getTotalMemory(numGPUs, freeCUDAMem, totalCUDAmem)) {
			qDebug() << numGPUs << "GPU(s) found. Total memory:" << totalCUDAmem << "bytes (" << totalCUDAmem / 1024 / 1024 << "MB). Available memory : " << freeCUDAMem << "bytes (" << freeCUDAMem / 1024 / 1024 << "MB).";
			size_t problemIfpTotalMem, problemIfpMaxMem, problemNfpTotalMem;
			RasterPackingProblem::getProblemGPUMemRequirements(problem, problemIfpTotalMem, problemIfpMaxMem, problemNfpTotalMem);
			qDebug() << problem.getInnerfitPolygonsCount() << "IFPs processed. Total IFP set size:" << problemIfpTotalMem << "bytes. (" << problemIfpTotalMem / 1024 / 1024 << "MB). Max size:" << problemIfpMaxMem << "bytes. (" << problemIfpMaxMem / 1024 / 1024 << "MB)."; // TOREMOVE
			qDebug() << problem.getNofitPolygonsCount() << "nfps processed. Total nfp set size:" << problemNfpTotalMem << "bytes. (" << problemNfpTotalMem / 1024 / 1024 << "MB)."; // TOREMOVE

			if (freeCUDAMem - problemIfpTotalMem - problemNfpTotalMem > 0) {
				size_t remainingCUDAMem = freeCUDAMem - problemIfpTotalMem - problemNfpTotalMem;
				//qDebug() << "Complete GPU allocation possible. Estimated free space after allocation:" << remainingCUDAMem << "bytes. (" << remainingCUDAMem / 1024 / 1024 << "MB).";
				if (QMessageBox::question(this, "Confirm CUDA allocation", "Complete GPU allocation possible. Estimated free space after allocation: " + QString::number(remainingCUDAMem / 1024 / 1024) + " MB. Do you want to allocate NFP memory?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
					loadGPU = true;
					CUDAPACKING::allocDeviceMaxIfp(problemIfpMaxMem);
				}
				
			}
			//else if (freeCUDAMem - problemIfpMaxMem - problemNfpTotalMem > 0) {
			//	size_t remainingCUDAMem = freeCUDAMem - problemIfpMaxMem - problemNfpTotalMem;
			//	qDebug() << "Partial GPU allocation possible. Estimated free space after allocation:" << remainingCUDAMem << "bytes. (" << remainingCUDAMem / 1024 / 1024 << "MB).";
			//}
		}
		else {
			qDebug() << "GPU not found.";
		}
		
		rasterProblem->load(problem, loadGPU);
		if (loadGPU) {  
			CUDAPACKING::getTotalMemory(numGPUs, freeCUDAMem, totalCUDAmem);
			qDebug() << "GPU memory allocated. Available memory" << freeCUDAMem / 1024 / 1024 << "MB.";
			ui->pushButton_11->setEnabled(true); ui->pushButton_21->setEnabled(true); ui->pushButton_22->setEnabled(true);
			ui->pushButton_23->setEnabled(true); ui->pushButton_24->setEnabled(true); ui->pushButton_25->setEnabled(true);
		}

        solution = RASTERVORONOIPACKING::RasterPackingSolution(rasterProblem->count());

        solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(rasterProblem));

        ui->graphicsView->setEnabled(true);
        ui->graphicsView->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
        ui->graphicsView->setStatusBar(ui->statusBar);
        ui->graphicsView->createGraphicItems(problem);
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
		ui->pushButton_27->setEnabled(true);
		
        ui->actionLoad_Zoomed_Problem->setEnabled(true);

        weightViewer.setWeights(solver->getGlsWeights(), solution.getNumItems());
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
    RASTERPREPROCESSING::PackingProblem problem;
    if(problem.load(fileName)) {
        rasterZoomedProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
        rasterZoomedProblem->load(problem);
        solver->setProblem(rasterZoomedProblem, true);

        ui->pushButton_16->setEnabled(true);
        ui->pushButton_17->setEnabled(true);
        ui->pushButton_18->setEnabled(true);
        ui->pushButton_19->setEnabled(true);
        ui->pushButton_20->setEnabled(true);
        ui->graphicsView->changeGridSize(rasterZoomedProblem->getScale());
        ui->doubleSpinBox->setSingleStep(1/rasterZoomedProblem->getScale());
        ui->doubleSpinBox_2->setSingleStep(1/rasterZoomedProblem->getScale());
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

//void updateOverlapMapsCachedInfo(RasterPackingSolution& newSolution, RasterPackingSolution& prevSolution, std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver) {
//	for (int curItemId = 0; curItemId < prevSolution.getNumItems(); curItemId++) {
//		if (newSolution.getPosition(curItemId) != prevSolution.getPosition(curItemId) || newSolution.getOrientation(curItemId) != prevSolution.getOrientation(curItemId)) {
//			qDebug() << "Cache info of item" << curItemId << "updated.";
//			solver->setCacheInfo(curItemId, true, newSolution.getPosition(curItemId), newSolution.getOrientation(curItemId));
//		}
//		else {
//			qDebug() << "Placement of item" << curItemId << "did not change.";
//			solver->setCacheInfo(curItemId, false, newSolution.getPosition(curItemId), newSolution.getOrientation(curItemId));
//		}
//	}
//}

void MainWindow::generateCurrentTotalOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, false);
	ui->graphicsView->showTotalOverlapMap(curMap);
	int milliseconds = myTimer.elapsed();
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::createRandomLayout() {
    solver->generateRandomSolution(solution);
    ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::translateCurrentToMinimumPosition() {
    qreal minVal;
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
    std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, false);
	QPoint minPos = solver->getMinimumOverlapPosition(curMap, minVal);
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
    qreal globalOverlap = solver->getGlobalOverlap(solution,overlaps);
    createOverlapMessageBox(globalOverlap, overlaps, this->rasterProblem->getScale());

//    QString log; QTextStream stream(&log); int linenum = 1;
//    stream <<  "Starting global overlap evaluation...\n";
//    stream <<  "Determining overlap for each item:\n";

//    int numNonColliding = 0;
//    for(int i = 0; i < rasterProblem->count(); i++) {
////        qreal curOverlap = solver->getTotalOverlap(i, solution);
////        totalOverlap += curOverlap;
//        qreal curOverlap = overlaps[i]/this->rasterProblem->getScale();
//        stream <<  "Item " << i << ": " << curOverlap;
//        if(qFuzzyCompare(1 + 0.0, 1 + curOverlap)) {
//            stream << " (no overlap)"; numNonColliding++;
//        }
//        stream << "\n";
//    }
//    stream <<  "Operation completed.\n";
//    if(qFuzzyCompare(1 + 0.0, 1 + globalOverlap))
//        stream <<  "Feasible layout! Zero overlap.\n";
//    else {
//        stream <<  "Unfeasible layout.\n";
//        stream <<  "Total overlap: " << globalOverlap/this->rasterProblem->getScale() << ". Average overlap: " << globalOverlap/this->rasterProblem->getScale()/(qreal)rasterProblem->count() << ".\n";
//        stream <<   numNonColliding << " non-colliding items and " << rasterProblem->count()-numNonColliding << " overlapped items.\n";
//    }

//    QMessageBox msgBox;
//    msgBox.setText("Global overlap evaluation completed successfully!");
//    msgBox.setInformativeText("The global overlap value is " + QString::number(globalOverlap/this->rasterProblem->getScale()));
//    msgBox.setDetailedText(log);
//    msgBox.exec();
}

void MainWindow::localSearch() {
    ui->graphicsView->getCurrentSolution(solution);
    QTime myTimer; myTimer.start();
    solver->performLocalSearch(solution, false, true);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);

	ui->pushButton_26->setEnabled(true);
	ui->pushButton_28->setEnabled(true);
	ui->pushButton_29->setEnabled(true);
}

void MainWindow::generateCurrentTotalGlsWeightedOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
    ui->graphicsView->showTotalOverlapMap(solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, true));
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::updateGlsWeightedOverlapMap() {
    ui->graphicsView->getCurrentSolution(solution);
    solver->updateWeights(solution);
    ui->statusBar->showMessage("Weights updated.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::resetGlsWeightedOverlapMap() {
    solver->resetWeights();
    ui->statusBar->showMessage("Weights reseted.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::translateCurrentToGlsWeightedMinimumPosition() {
    qreal minVal;
    ui->graphicsView->getCurrentSolution(solution);
    int itemId = ui->graphicsView->getCurrentItemId();
    std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMap(itemId, solution.getOrientation(itemId), solution, true);
    solution.setPosition(itemId, solver->getMinimumOverlapPosition(curMap, minVal));
    ui->graphicsView->setCurrentSolution(solution);
    ui->graphicsView->showTotalOverlapMap(curMap);
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::glsWeightedlocalSearch() {
//    ui->graphicsView->getCurrentSolution(solution);
//    QTime myTimer; myTimer.start();
//    solver->performGlsWeightedLocalSearch(solution);
//    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
//    ui->graphicsView->setCurrentSolution(solution);
//    weightViewer.updateImage();
//    weightViewer.show();
    ui->graphicsView->getCurrentSolution(solution);
    QTime myTimer; myTimer.start();
    solver->performLocalSearch(solution, true, true);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution);
    weightViewer.updateImage();
    weightViewer.show();

	ui->pushButton_30->setEnabled(true);
	ui->pushButton_31->setEnabled(true);
}


void MainWindow::executePacking() {
    weightViewer.updateImage();
    weightViewer.show();

    // Resize container
    qreal lenght = runConfig.getLenght();
    int scaledWidth = qRound(lenght*ui->graphicsView->getScale());
    solver->setContainerWidth(scaledWidth);
    ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());

    if(runConfig.getMetaheuristic() == 2) solver->switchProblem(true);
    if(runConfig.getInitialSolution() == 1)  solver->generateRandomSolution(solution);
    else if(runConfig.getMetaheuristic() == 2) ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    else ui->graphicsView->getCurrentSolution(solution);
    solver->switchProblem(false);
    runThread.setInitialSolution(solution);
	runThread.setParameters(runConfig.getMaxWorse(), runConfig.getMetaheuristic(), runConfig.getMaxSeconds(), runConfig.getUseCUDA(), runConfig.getCacheMaps());
    runThread.setSolver(solver);
    if(runConfig.getMetaheuristic() == 0 || runConfig.getMetaheuristic() == 1) runThread.setScale(this->rasterProblem->getScale());
    else if(runConfig.getMetaheuristic() == 2) runThread.setScale(this->rasterZoomedProblem->getScale());
    runThread.start();
}

void MainWindow::changeContainerWidth() {
    // FIXME: Create custom dialog
    qreal lenght = QInputDialog::getDouble(this, "New container lenght", "Lenght:", (qreal)solver->getCurrentWidth()/ui->graphicsView->getScale(), 0, (qreal)this->rasterProblem->getContainerWidth()/ui->graphicsView->getScale(),2);
    int scaledWidth = qRound(lenght*ui->graphicsView->getScale());
    solver->setContainerWidth(scaledWidth);
    ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());
}

void MainWindow::showZoomedMap() {
    int zoomSquareSize = 3*qRound(this->rasterZoomedProblem->getScale()/this->rasterProblem->getScale());
    ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    int itemId = ui->graphicsView->getCurrentItemId();
    solver->switchProblem(true);
    std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = solver->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution, true);
   solver-> switchProblem(false);
    QPixmap zoomImage = QPixmap::fromImage(zoomMap->getImage());
    zoomedMapViewer.setImage(zoomImage);
    zoomedMapViewer.show();
}

void MainWindow::translateCurrentToMinimumZoomedPosition() {
    qreal minVal;
    int zoomSquareSize = 3*qRound(this->rasterZoomedProblem->getScale()/this->rasterProblem->getScale());
    ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    int itemId = ui->graphicsView->getCurrentItemId();
    solver->switchProblem(true);
    std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> zoomMap = solver->getRectTotalOverlapMap(itemId, solution.getOrientation(itemId), solution.getPosition(itemId), zoomSquareSize, zoomSquareSize, solution, true);
    solver->switchProblem(false);
    solution.setPosition(itemId, solver->getMinimumOverlapPosition(zoomMap, minVal));
    ui->graphicsView->setCurrentSolution(solution, this->rasterZoomedProblem->getScale());

    QPixmap zoomImage = QPixmap::fromImage(zoomMap->getImage());
    zoomedMapViewer.setImage(zoomImage);
    zoomedMapViewer.show();
}

void MainWindow::showZoomedGlobalOverlap() {
    ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    QVector<qreal> overlaps;
    solver->switchProblem(true);
    qreal globalOverlap = solver->getGlobalOverlap(solution,overlaps);
    solver->switchProblem(false);
    createOverlapMessageBox(globalOverlap, overlaps, this->rasterZoomedProblem->getScale());
}

void MainWindow::updateZoomedGlsWeights() {
    ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    solver->switchProblem(true);
    solver->updateWeights(solution);
    solver->switchProblem(false);
    ui->statusBar->showMessage("Weights updated.");
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::zoomedlocalSearch() {
    ui->graphicsView->getCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    QTime myTimer; myTimer.start();
    solver->performTwoLevelLocalSearch(solution, true);
    ui->statusBar->showMessage( "Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed()/1000.0) + "seconds");
    ui->graphicsView->setCurrentSolution(solution, this->rasterZoomedProblem->getScale());
    weightViewer.updateImage();
    weightViewer.show();
}

void MainWindow::showExecutionStatus(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale) {
    statusBar()->showMessage("Iteration: " + QString::number(totalItNum) + " (" + QString::number(worseSolutionsCount) +
                             "). Overlap: " +  QString::number(curOverlap/scale) + ". Minimum overlap: " + QString::number(minOverlap/scale) +
                             ". Time elapsed: " + QString::number(elapsed) + " secs.");
}

void MainWindow::showExecutionFinishedStatus(int totalItNum, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale) {
    statusBar()->showMessage("Finished. Total iterations: " + QString::number(totalItNum) +
                             ".Minimum overlap: " + QString::number(minOverlap/scale) +
                             ". Elapsed time: " + QString::number(elapsed));
    if(qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) {
        ui->graphicsView->getCurrentSolution(solution, scale);
//        qDebug() << solution;
    }
}

void MainWindow::saveSolution() {
    QString  fileName = QFileDialog::getSaveFileName(this, tr("Save solution"), "", tr("Modified ESICUP Files (*.xml)"));

	QFile file(fileName);
	if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qCritical() << "Error: Cannot open file"
                    << ": " << qPrintable(file.errorString());
        return;
    }

	QXmlStreamWriter stream(&file);
	stream.setAutoFormatting(true);
	stream.writeStartDocument();
	stream.writeStartElement("solution");
	for(int itemId = 0; itemId < this->rasterProblem->count(); itemId++) {
		std::shared_ptr<RasterPackingItem> curItem = this->rasterProblem->getItem(itemId);
		stream.writeStartElement("placement");
		stream.writeAttribute("boardNumber", "1");
		stream.writeAttribute("x", QString::number(this->solution.getPosition(itemId).x()/this->rasterProblem->getScale()));
		stream.writeAttribute("y", QString::number(this->solution.getPosition(itemId).y()/this->rasterProblem->getScale()));
		stream.writeAttribute("idBoard", this->rasterProblem->getContainerName());
		stream.writeAttribute("idPiece", "piece" + QString::number(this->rasterProblem->getItemType(itemId))); // FIXME: Change to Name
		stream.writeAttribute("angle", QString::number(this->rasterProblem->getItem(itemId)->getAngleValue(this->solution.getOrientation(itemId))));
		stream.writeAttribute("mirror", "none");
		stream.writeEndElement(); // placement
	}
	stream.writeTextElement("length", QString::number(solver->getCurrentWidth()/this->rasterProblem->getScale()));
	stream.writeEndElement(); // solution
	stream.writeEndDocument();

	file.close();
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

    xml.setDevice(&file);
    int itemId = 0;
    while (!xml.atEnd()) {
        xml.readNext();

        if(xml.name()=="placement" && xml.tokenType() == QXmlStreamReader::StartElement) {
            int posX = qRound(this->rasterProblem->getScale()*xml.attributes().value("x").toFloat());
            int posY = qRound(this->rasterProblem->getScale()*xml.attributes().value("y").toFloat());
            solution.setPosition(itemId, QPoint(posX,posY));
            unsigned int angleId = 0;
            for(; angleId < this->rasterProblem->getItem(itemId)->getAngleCount(); angleId++)
                if(this->rasterProblem->getItem(itemId)->getAngleValue(angleId) == xml.attributes().value("angle").toInt())
                    break;
            solution.setOrientation(itemId, angleId);
//            qDebug() << itemId << xml.attributes().value("x").toFloat() << xml.attributes().value("y").toFloat() << xml.attributes().value("angle").toInt();
            itemId++;
        }

        if(xml.name()=="length"  && xml.tokenType() == QXmlStreamReader::StartElement) {
            int scaledWidth = qRound(xml.readElementText().toFloat()*this->rasterProblem->getScale());
            solver->setContainerWidth(scaledWidth);
            ui->graphicsView->recreateContainerGraphics(solver->getCurrentWidth());
        }

        if(xml.name()=="solution" && xml.tokenType() == QXmlStreamReader::EndElement) break;
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

void MainWindow::generateCurrentTotalOverlapMapGPU() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapGPU(itemId, solution.getOrientation(itemId), solution, false);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::translateCurrentToMinimumPositionGPU() {
	qreal minVal;
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	QPoint newPos = solver->getMinimumOverlapPositionGPU(itemId, solution.getOrientation(itemId), solution, minVal, false);
	solution.setPosition(itemId, newPos);
	ui->graphicsView->setCurrentSolution(solution);
	int milliseconds = myTimer.elapsed();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapGPU(itemId, solution.getOrientation(itemId), solution, false);
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Minimum position determined. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::localSearchGPU() {
	ui->graphicsView->getCurrentSolution(solution);
	QTime myTimer; myTimer.start();
	solver->performLocalSearchGPU(solution, false);
	ui->statusBar->showMessage("Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed() / 1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::generateCurrentTotalGlsWeightedOverlapMapGPU() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapGPU(itemId, solution.getOrientation(itemId), solution, true);
	int milliseconds = myTimer.elapsed();
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
	weightViewer.updateImage();
	weightViewer.show();
}

void MainWindow::translateCurrentToGlsWeightedMinimumPositionGPU() {
	qreal minVal;
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	QPoint newPos = solver->getMinimumOverlapPositionGPU(itemId, solution.getOrientation(itemId), solution, minVal, true);
	solution.setPosition(itemId, newPos);
	ui->graphicsView->setCurrentSolution(solution);
	int milliseconds = myTimer.elapsed();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapGPU(itemId, solution.getOrientation(itemId), solution, true);
	ui->graphicsView->showTotalOverlapMap(curMap);
	ui->statusBar->showMessage("Minimum position determined. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
	weightViewer.updateImage();
	weightViewer.show();
}

void MainWindow::glsWeightedlocalSearchGPU() {
	ui->graphicsView->getCurrentSolution(solution);
	QTime myTimer; myTimer.start();
	solver->performLocalSearchGPU(solution, true);
	ui->statusBar->showMessage("Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed() / 1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
	weightViewer.updateImage();
	weightViewer.show();
}

void MainWindow::localSearchCache() {
	ui->graphicsView->getCurrentSolution(solution);
	QTime myTimer; myTimer.start();
	solver->performLocalSearchwithCache(solution, false);
	ui->statusBar->showMessage("Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed() / 1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
}

void MainWindow::printCacheInfo() {
	int itemId = ui->graphicsView->getCurrentItemId();
	solver->printCompleteCacheInfo(itemId, solution.getOrientation(itemId));
}

void MainWindow::generateCurrentTotalOverlapMapCache() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapwithCache(itemId, solution.getOrientation(itemId), solution, false);
	ui->graphicsView->showTotalOverlapMap(curMap);
	int milliseconds = myTimer.elapsed();
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::printGlsWeightedCacheInfo() {
	int itemId = ui->graphicsView->getCurrentItemId();
	solver->printCompleteCacheInfo(itemId, solution.getOrientation(itemId), true);
}

void MainWindow::generateCurrentTotalGlsWeightedOverlapMapCache() {
	ui->graphicsView->getCurrentSolution(solution);
	int itemId = ui->graphicsView->getCurrentItemId();
	QTime myTimer; myTimer.start();
	std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> curMap = solver->getTotalOverlapMapwithCache(itemId, solution.getOrientation(itemId), solution, true);
	ui->graphicsView->showTotalOverlapMap(curMap);
	int milliseconds = myTimer.elapsed();
	ui->statusBar->showMessage("Total overlap map created. Elapsed Time: " + QString::number(milliseconds) + " miliseconds");
}

void MainWindow::glsWeightedlocalSearchCache() {
	ui->graphicsView->getCurrentSolution(solution);
	QTime myTimer; myTimer.start();
	solver->performLocalSearchwithCache(solution, true);
	ui->statusBar->showMessage("Local search concluded. Elapsed Time: " + QString::number(myTimer.elapsed() / 1000.0) + "seconds");
	ui->graphicsView->setCurrentSolution(solution);
}