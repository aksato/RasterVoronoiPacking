#include "consolepackingloader.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "packingproblem.h"
#include "cuda/gpuinfo.h"
#include <QDir>
#include <iostream>
#include <iomanip>

ConsolePackingLoader::ConsolePackingLoader(QObject *parent) {
	numProcesses = 0;
}

bool ConsolePackingLoader::loadInputFile(QString inputFilePath, std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> problem, bool &loadGPU) {
	RASTERPREPROCESSING::PackingProblem preProblem;
	if (!preProblem.load(inputFilePath)) {
		qCritical("Could not open file '%s'!", qPrintable(inputFilePath));
		return false;
	}

	// GPU problem loading
	if (loadGPU) {
		GpuMemoryRequirements gpuMemReq;
		int numGPUs;  size_t freeCUDAMem, totalCUDAmem;

		RASTERVORONOIPACKING::RasterPackingProblem::getProblemGPUMemRequirements(preProblem, gpuMemReq.totalIfpMemory, gpuMemReq.maxSingleIfpMemory, gpuMemReq.totalNfpMemory);
		if (CUDAPACKING::getTotalMemory(numGPUs, freeCUDAMem, totalCUDAmem) && freeCUDAMem - gpuMemReq.totalIfpMemory - gpuMemReq.totalNfpMemory > 0) CUDAPACKING::allocDeviceMaxIfp(gpuMemReq.maxSingleIfpMemory);
		else loadGPU = false;
	}

	problem->load(preProblem, loadGPU);
	return true;
}

void ConsolePackingLoader::setParameters(QString inputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams) {
	// FIXME: Is it necessary?
	algorithmParamsBackup.Copy(algorithmParams); this->outputTXTFile = outputTXTFile; this->outputXMLFile = outputXMLFile;

	// Create solution and problem 
	RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERPREPROCESSING::PackingProblem preProblem;
	GpuMemoryRequirements gpuMemReq;
	qDebug() << "Program execution started.";

	// Load input file
	qDebug() << "Loading problem file...";
	qDebug() << "Input file:" << inputFilePath;
	QString originalPath = QDir::currentPath();
	QDir::setCurrent(QFileInfo(inputFilePath).absolutePath());
	problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
	bool loadGPU = algorithmParams.isGpuProcessing(); loadInputFile(inputFilePath, problem, loadGPU); algorithmParams.setGpuProcessing(loadGPU);
	QDir::setCurrent(originalPath);
	qDebug() << "Problem file read successfully";
}

// Create problem objects and initial solution using given parameters
void ConsolePackingLoader::setParameters(QString inputFilePath, QString zoomedInputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams) {
	// FIXME: Is it necessary?
	algorithmParamsBackup.Copy(algorithmParams); this->outputTXTFile = outputTXTFile; this->outputXMLFile = outputXMLFile;

	// Create solution and problem 
	RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERPREPROCESSING::PackingProblem preProblem;
	GpuMemoryRequirements gpuMemReq;
	qDebug() << "Program execution started.";

	// Load input files
	qDebug() << "Loading problem file...";
	qDebug() << "Input file:" << inputFilePath;
	qDebug() << "Zoom Input file:" << zoomedInputFilePath;
	QString originalPath = QDir::currentPath();
	QDir::setCurrent(QFileInfo(inputFilePath).absolutePath());
	problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
	bool loadGPU = algorithmParams.isGpuProcessing(); loadInputFile(inputFilePath, problem, loadGPU); algorithmParams.setGpuProcessing(loadGPU);
	QDir::setCurrent(QFileInfo(zoomedInputFilePath).absolutePath());
	zoomProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
	loadGPU = false;  loadInputFile(zoomedInputFilePath, zoomProblem, loadGPU);
	QDir::setCurrent(originalPath);
	qDebug() << "Problem file read successfully";
}

void ConsolePackingLoader::run() {
	// Create solver object
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(problem));
	if (algorithmParamsBackup.isDoubleResolution()) solver->setProblem(zoomProblem, true);

	// Create initial solution
	RASTERVORONOIPACKING::RasterPackingSolution solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count(), algorithmParamsBackup.isGpuProcessing());
	qreal length;
	if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) {
		length = algorithmParamsBackup.getInitialLenght();
		int scaledWidth = qRound(length*problem->getScale());
		solver->setContainerWidth(scaledWidth);
		solver->generateRandomSolution(solution, algorithmParamsBackup);
	}
	else {
		qCritical() << "Returning. Initial solution method unavailable:" << algorithmParamsBackup.getInitialSolMethod();
		return;
	}

	// Configure packer object
	std::shared_ptr<PackingThread> threadedPacker = std::shared_ptr<PackingThread>(new PackingThread);
	threadVector.push_back(threadedPacker);
	connect(&*threadedPacker, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(printExecutionStatus(int, int, int, qreal, qreal, qreal)));
	connect(&*threadedPacker, SIGNAL(minimumLenghtUpdated(int,int,qreal,uint)), SLOT(saveMinimumResult(int,int,qreal,uint)));
	qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	connect(&*threadedPacker, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)), SLOT(saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)));
	connect(&*threadedPacker, SIGNAL(finished()), SLOT(threadFinished()));

	threadedPacker->setInitialSolution(solution);
	threadedPacker->setParameters(algorithmParamsBackup);
	threadedPacker->setSolver(solver);

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	if (!algorithmParamsBackup.isDoubleResolution()) qDebug() << "Problem Scale:" << problem->getScale();
	else qDebug() << "Problem Scale:" << zoomProblem->getScale() << ". Auxiliary problem scale:" << problem->getScale();
	qDebug() << "Length:" << length;
	qDebug() << "Solver method:" << algorithmParamsBackup.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParamsBackup.getInitialSolMethod();
	if (!algorithmParamsBackup.isFixedLength()) qDebug() << "Strip packing version";
	if (algorithmParamsBackup.isGpuProcessing()) qDebug() << "Using GPU to process maps";
	qDebug() << "Solver parameters: Nmo =" << algorithmParamsBackup.getNmo() << "; Time Limit:" << algorithmParamsBackup.getTimeLimit();

	numProcesses++;
	// Run!
	threadedPacker->start();
}

void ConsolePackingLoader::printExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed) {
	if (!algorithmParamsBackup.isDoubleResolution())
		std::cout << std::fixed << std::setprecision(2) << "\r" << "L: " << curLength / problem->getScale() <<
		". It: " << totalItNum << " (" << worseSolutionsCount << "). Min overlap: " << minOverlap / problem->getScale() << ". Time: " << elapsed << " s.";
	else
		std::cout << std::fixed << std::setprecision(2) << "\r" << "L: " << curLength / problem->getScale() <<
		". It: " << totalItNum << " (" << worseSolutionsCount << "). Min overlap: " << minOverlap / zoomProblem->getScale() << ". Time: " << elapsed << " s.";
}

void ConsolePackingLoader::saveMinimumResult(int minLength, int totalItNum, qreal elapsed, uint threadSeed) {
	std::cout << "\n" << "New minimum length obtained: " << minLength / problem->getScale() << ". Elapsed time: " << elapsed << " secs. Seed = " << threadSeed << "\n";

	QFile file(outputTXTFile);
	if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << outputTXTFile << ": " << qPrintable(file.errorString());
	QTextStream out(&file);
	if(!algorithmParamsBackup.isDoubleResolution())
		out << problem->getScale() << " - " << minLength / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else
		out << problem->getScale() << " " << zoomProblem->getScale() << " " << minLength / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	file.close();
}

void ConsolePackingLoader::saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, int length, int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime, uint seed) {
	if (algorithmParamsBackup.isFixedLength()) 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << totalTime;
	else 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum length =" << length / problem->getScale() << ". Elapsed time:" << totalTime;

	// Save Layout in XML file
	RASTERVORONOIPACKING::RasterPackingSolution solution(bestSolution.getNumItems());
	for (int i = 0; i < bestSolution.getNumItems(); i++) {
		solution.setPosition(i, bestSolution.getPosition(i));
		solution.setOrientation(i, bestSolution.getOrientation(i));
	}
	if (!algorithmParamsBackup.isDoubleResolution()) solution.save(outputXMLFile, problem, length, true, seed);
	else solution.save(outputXMLFile, zoomProblem, length, true, seed);

	// Print fixed container final result to file
	if (algorithmParamsBackup.isFixedLength()) {
		QFile file(outputTXTFile);
		if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << outputTXTFile << ": " << qPrintable(file.errorString());
		QTextStream out(&file);
		if (!algorithmParamsBackup.isDoubleResolution())
			out << problem->getScale() << " - " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		else
			out << problem->getScale() << " " << zoomProblem->getScale() << " " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		file.close();
	}
}

void ConsolePackingLoader::threadFinished() {
	numProcesses--;
	if (numProcesses == 0) emit quitApp();
}