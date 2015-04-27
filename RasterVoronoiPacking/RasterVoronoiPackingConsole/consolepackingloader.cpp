#include "consolepackingloader.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "packingproblem.h"
#include "cuda/gpuinfo.h"
#include <QDir>
#include <iostream>
#include <iomanip>

ConsolePackingLoader::ConsolePackingLoader(QObject *parent) {
	connect(&singleThreadedPacker, SIGNAL(statusUpdated(int, int, qreal, qreal, qreal, qreal, qreal, int, int)), this, SLOT(printExecutionStatus(int, int, qreal, qreal, qreal, qreal, qreal, int, int)));
	connect(&singleThreadedPacker, SIGNAL(minimumLenghtUpdated(int, qreal, qreal, int, qreal, uint)), SLOT(saveMinimumResult(int, qreal, qreal, int, qreal, uint)));
	connect(&singleThreadedPacker, SIGNAL(finishedExecution(int, qreal, qreal, qreal, qreal, qreal, int, uint)), SLOT(saveFinalResult(int, qreal, qreal, qreal, qreal, qreal, int, uint)));
	qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	connect(&singleThreadedPacker, SIGNAL(finalSolutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution, qreal, qreal, uint)), SLOT(saveFinalLayout(const RASTERVORONOIPACKING::RasterPackingSolution, qreal, qreal, uint)));
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
	
	// Create solver object
	solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(problem));

	// Create initial solution
	solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count(), algorithmParams.isGpuProcessing());
	qreal length;
	if (algorithmParams.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) {
		length = algorithmParams.getInitialLenght();
		int scaledWidth = qRound(length*problem->getScale());
		solver->setContainerWidth(scaledWidth);
		solver->generateRandomSolution(solution, algorithmParams);
	}
	else {
		qCritical() << "Returning. Initial solution method unavailable:" << algorithmParams.getInitialSolMethod();
		return;
	}

	// Configure packer object
	qreal scale = problem->getScale();
	singleThreadedPacker.setInitialSolution(solution);
	singleThreadedPacker.setParameters(algorithmParams);
	singleThreadedPacker.setSolver(solver);
	singleThreadedPacker.setScale(scale);

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	qDebug() << "Problem Scale:" << scale;
	qDebug() << "Length:" << length;
	qDebug() << "Solver method:" << algorithmParams.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParams.getInitialSolMethod();
	if (!algorithmParams.isFixedLength()) qDebug() << "Strip packing version";
	if (algorithmParams.isGpuProcessing()) qDebug() << "Using GPU to process maps";
	qDebug() << "Solver parameters: Nmo =" << algorithmParams.getNmo() << "; Time Limit:" << algorithmParams.getTimeLimit();
}

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

	// Create solver object
	solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(problem));
	solver->setProblem(zoomProblem, true);

	// Create initial solution
	solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count(), algorithmParams.isGpuProcessing());
	qreal length;
	if (algorithmParams.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) {
		length = algorithmParams.getInitialLenght();
		int scaledWidth = qRound(length*problem->getScale());
		solver->setContainerWidth(scaledWidth);
		solver->generateRandomSolution(solution, algorithmParams);
	}
	else {
		qCritical() << "Returning. Initial solution method unavailable:" << algorithmParams.getInitialSolMethod();
		return;
	}

	// Configure packer object
	singleThreadedPacker.setInitialSolution(solution);
	singleThreadedPacker.setParameters(algorithmParams);
	singleThreadedPacker.setSolver(solver);
	singleThreadedPacker.setScale(zoomProblem->getScale(), problem->getScale());

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	qDebug() << "Problem Scale:" << zoomProblem->getScale() << ". Auxiliary problem scale:" << problem->getScale();
	qDebug() << "Length:" << length;
	qDebug() << "Solver method:" << algorithmParams.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParams.getInitialSolMethod();
	if (!algorithmParams.isFixedLength()) qDebug() << "Strip packing version";
	if (algorithmParams.isGpuProcessing()) qDebug() << "Using GPU to process maps";
	qDebug() << "Solver parameters: Nmo =" << algorithmParams.getNmo() << "; Time Limit:" << algorithmParams.getTimeLimit();
}

void ConsolePackingLoader::run() {
	// Run!
	singleThreadedPacker.run();
}

void ConsolePackingLoader::printExecutionStatus(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, qreal zoomscale, int curLength, int minLength) {
	std::cout << std::fixed << std::setprecision(2) << "\r" << "Lenght: " << curLength / scale << " (Min: " << minLength / scale << "). It: " << totalItNum << " (" << worseSolutionsCount << "). Min overlap: " << minOverlap / scale << ". Time: " << elapsed << " secs.";
}

void ConsolePackingLoader::saveMinimumResult(int minLength, qreal scale, qreal zoomScale, int totalItNum, qreal elapsed, uint seed) {
	std::cout << "\n" << "New minimum length obtained: " << minLength / scale << ". Elapsed time: " << elapsed << " secs.\n";

	QFile file(outputTXTFile);
	if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << outputTXTFile << ": " << qPrintable(file.errorString());
	QTextStream out(&file);
	if(!algorithmParamsBackup.isDoubleResolution())
		out << scale << " - " << minLength / scale << " " << totalItNum << " " << elapsed << " " << elapsed / totalItNum << " " << seed << "\n";
	else
		out << scale << " " << zoomScale << " " << minLength / scale << " " << totalItNum << " " << elapsed << " " << elapsed / totalItNum << " " << seed << "\n";
	file.close();
}

void ConsolePackingLoader::saveFinalResult(int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime, qreal scale, qreal zoomscale, int length, uint seed) {
	if (algorithmParamsBackup.isFixedLength()) qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << totalTime;
	else qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum length =" << length / scale << ". Elapsed time:" << totalTime;

	if (!algorithmParamsBackup.isFixedLength()) return;
	QFile file(outputTXTFile);
	if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << outputTXTFile << ": " << qPrintable(file.errorString());
	QTextStream out(&file);
	if (!algorithmParamsBackup.isDoubleResolution())
		out << scale << " - " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
	else
		out << scale << " " << zoomscale << " " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
	file.close();
}

void ConsolePackingLoader::saveFinalLayout(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, qreal scale, qreal length, uint seed) {
	RASTERVORONOIPACKING::RasterPackingSolution solution(bestSolution.getNumItems());
	for(int i = 0; i < bestSolution.getNumItems(); i++) {
		solution.setPosition(i, bestSolution.getPosition(i));
		solution.setOrientation(i, bestSolution.getOrientation(i));
	}
	if (!algorithmParamsBackup.isDoubleResolution()) solution.save(outputXMLFile, problem, length, true, seed);
	else solution.save(outputXMLFile, zoomProblem, length, true, seed);
}