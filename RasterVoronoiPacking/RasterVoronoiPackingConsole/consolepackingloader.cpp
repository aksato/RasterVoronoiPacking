#include "consolepackingloader.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolvergls.h"
#include "raster/rasterstrippackingsolverdoublegls.h"
#include "raster/packingclusterthread.h"
#include "packingproblem.h"
#include <QDir>
#include <QXmlStreamWriter>
#include <iostream>
#include <iomanip>

ConsolePackingLoader::ConsolePackingLoader(QObject *parent) {
	numProcesses = 0;
}

bool ConsolePackingLoader::loadInputFile(QString inputFilePath, std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> *problem) {
	RASTERPACKING::PackingProblem preProblem;
	if (!preProblem.load(inputFilePath)) {
		qCritical("Could not open file '%s'!", qPrintable(inputFilePath));
		return false;
	}
	if (preProblem.loadClusterInfo(inputFilePath)) {
		*problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem>(new RASTERVORONOIPACKING::RasterPackingClusterProblem);
		qDebug() << "Cluster problem detected.";
	}
	else {
		*problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
		if (algorithmParamsBackup.getClusterFactor() > 0) {
			algorithmParamsBackup.setClusterFactor(-1.0);
			qWarning() << "Cluster problem not detected, ignoring cluster factor value.";
		}
	}
	(*problem)->load(preProblem);
	return true;
}

void ConsolePackingLoader::setParameters(QString inputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams, bool appendSeed) {
	// FIXME: Is it necessary?
	algorithmParamsBackup.Copy(algorithmParams); this->outputTXTFile = outputTXTFile; this->outputXMLFile = outputXMLFile; this->appendSeedToOutputFiles = appendSeed;

	// Create solution and problem 
	RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERPACKING::PackingProblem preProblem;
	GpuMemoryRequirements gpuMemReq;
	qDebug() << "Program execution started.";

	// Load input file
	qDebug() << "Loading problem file...";
	qDebug() << "Input file:" << inputFilePath;
	QString originalPath = QDir::currentPath();
	QDir::setCurrent(QFileInfo(inputFilePath).absolutePath());
	loadInputFile(inputFilePath, &problem);
	QDir::setCurrent(originalPath);
	qDebug() << "Problem file read successfully";
}

// Create problem objects and initial solution using given parameters
void ConsolePackingLoader::setParameters(QString inputFilePath, QString zoomedInputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams, bool appendSeed) {
	// FIXME: Is it necessary?
	algorithmParamsBackup.Copy(algorithmParams); this->outputTXTFile = outputTXTFile; this->outputXMLFile = outputXMLFile; this->appendSeedToOutputFiles = appendSeed;

	// Create solution and problem 
	RASTERVORONOIPACKING::RasterPackingSolution solution;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERPACKING::PackingProblem preProblem;
	GpuMemoryRequirements gpuMemReq;
	qDebug() << "Program execution started.";

	// Load input files
	qDebug() << "Loading problem file...";
	qDebug() << "Input file:" << inputFilePath;
	qDebug() << "Zoom Input file:" << zoomedInputFilePath;
	QString originalPath = QDir::currentPath();
	QDir::setCurrent(QFileInfo(inputFilePath).absolutePath());
	loadInputFile(inputFilePath, &problem);
	QDir::setCurrent(QFileInfo(zoomedInputFilePath).absolutePath());
	loadInputFile(zoomedInputFilePath, &zoomProblem);
	QDir::setCurrent(originalPath);
	qDebug() << "Problem file read successfully";
}

void ConsolePackingLoader::run() {
	std::shared_ptr<PackingThread> threadedPacker;

	// Create solver object
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	bool clusterExecution = algorithmParamsBackup.getClusterFactor() > 0;
	if (!clusterExecution) {
		if (!algorithmParamsBackup.isDoubleResolution()) solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(problem));
		else solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(problem, zoomProblem));
		threadedPacker = std::shared_ptr<PackingThread>(new PackingThread);
		threadedPacker->setSolver(solver);
	}
	else {
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> clusterSolverGls;
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> originalSolverGls;
		// Get pointer to cluster problems
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(problem);
		if (algorithmParamsBackup.isDoubleResolution()) {
			// Create new problems
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterSearchProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(zoomProblem);
			clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS(clusterProblem, clusterSearchProblem));
			originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(clusterProblem->getOriginalProblem(), clusterSearchProblem->getOriginalProblem()));
		}
		else {
			// Create new problem
			clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS(clusterProblem));
			originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(clusterProblem->getOriginalProblem()));
		}
		// Configure Thread
		threadedPacker = std::shared_ptr<PackingClusterThread>(new PackingClusterThread);
		std::shared_ptr<PackingClusterThread> threadedClusterPacker = std::dynamic_pointer_cast<PackingClusterThread>(threadedPacker);
		threadedClusterPacker->setSolver(originalSolverGls, clusterSolverGls);
		connect(&*threadedClusterPacker, SIGNAL(unclustered(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)), this, SLOT(updateUnclusteredProblem(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)));
	}
	
	// Resize container to initial length
	RASTERVORONOIPACKING::RasterPackingSolution solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count());
	qreal length;
	if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) {
		length = algorithmParamsBackup.getInitialLenght();
		int initialWidth = qRound(length*problem->getScale());
		solver->setContainerWidth(initialWidth, solution, algorithmParamsBackup);
	}

	// Configure packer object
	threadVector.push_back(threadedPacker);
	connect(&*threadedPacker, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(printExecutionStatus(int, int, int, qreal, qreal, qreal)));
	qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	connect(&*threadedPacker, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)), SLOT(saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, uint)));
	connect(&*threadedPacker, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)), SLOT(saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution, int, int, qreal, qreal, qreal, uint)));
	connect(&*threadedPacker, SIGNAL(finished()), SLOT(threadFinished()));
	threadedPacker->setParameters(algorithmParamsBackup);

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	if (!algorithmParamsBackup.isDoubleResolution()) qDebug() << "Problem Scale:" << problem->getScale();
	else qDebug() << "Problem Scale:" << zoomProblem->getScale() << ". Auxiliary problem scale:" << problem->getScale();
	if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) qDebug() << "Length:" << length;
	qDebug() << "Solver method:" << algorithmParamsBackup.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParamsBackup.getInitialSolMethod();
	qDebug() << "Minimum overlap placement heuristic:" << algorithmParamsBackup.getPlacementCriteria();
	if (!algorithmParamsBackup.isFixedLength()) qDebug() << "Strip packing version";
	qDebug() << "Solver parameters: Nmo =" << algorithmParamsBackup.getNmo() << "; Time Limit:" << algorithmParamsBackup.getTimeLimit();
	if (algorithmParamsBackup.getClusterFactor() > 0) qDebug() << "Cluster factor:" << algorithmParamsBackup.getClusterFactor();
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

QString getSeedAppendedString(QString originalString, uint seed) {
	QFileInfo fileInfo(originalString);
	QString newFileName = fileInfo.baseName() + "_" + QString::number(seed) + "." + fileInfo.suffix();
	return QDir(QFileInfo(originalString).path()).filePath(newFileName);
}

void ConsolePackingLoader::writeNewLength(int length, int totalItNum, qreal elapsed, uint threadSeed) {
	QString processedOutputTXTFile = appendSeedToOutputFiles ? getSeedAppendedString(outputTXTFile, threadSeed) : outputTXTFile;
	QFile file(processedOutputTXTFile);
	if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << outputTXTFile << ": " << qPrintable(file.errorString());
	QTextStream out(&file);
	if (!algorithmParamsBackup.isDoubleResolution())
		out << problem->getScale() << " - " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else
		out << problem->getScale() << " " << zoomProblem->getScale() << " " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	file.close();
}

void ConsolePackingLoader::saveXMLSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, uint seed) {
	qreal realLength = length / problem->getScale();
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> newSolution(new RASTERVORONOIPACKING::RasterPackingSolution(solution.getNumItems()));
	for (int i = 0; i < solution.getNumItems(); i++) {
		newSolution->setPosition(i, solution.getPosition(i));
		newSolution->setOrientation(i, solution.getOrientation(i));
	}
	solutionsCompilation.push_back(QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, qreal>(newSolution, realLength));
}

void ConsolePackingLoader::saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal elapsed, uint threadSeed) {
	std::cout << "\n" << "New minimum length obtained: " << minLength / problem->getScale() << ". Elapsed time: " << elapsed << " secs. Seed = " << threadSeed << "\n";
	writeNewLength(minLength, totalItNum, elapsed, threadSeed);
	saveXMLSolution(solution, minLength, threadSeed);
}

void ConsolePackingLoader::updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed) {
	// Print to console
	std::cout << "\n" << "Undoing the initial clusters, returning to original problem. Current length:" << length / problem->getScale() << ". Elapsed time: " << elapsed << " secs" << "\n";
}

void ConsolePackingLoader::saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, int length, int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime, uint seed) {
	if (algorithmParamsBackup.isFixedLength()) 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << totalTime;
	else 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum length =" << length / problem->getScale() << ". Elapsed time:" << totalTime;
	writeNewLength(length, totalIt, totalTime, seed);

	// Determine output file names
	QString processedOutputTXTFile = appendSeedToOutputFiles ? getSeedAppendedString(outputTXTFile, seed) : outputTXTFile;
	QString processedOutputXMLFile = appendSeedToOutputFiles ? getSeedAppendedString(outputXMLFile, seed) : outputXMLFile;

	// Print fixed container final result to file
	if (algorithmParamsBackup.isFixedLength()) {
		QFile file(processedOutputTXTFile);
		if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
		QTextStream out(&file);
		if (!algorithmParamsBackup.isDoubleResolution())
			out << problem->getScale() << " - " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		else
			out << problem->getScale() << " " << zoomProblem->getScale() << " " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		file.close();
	}

	// Print solutions collection xml file
	QFile file(processedOutputXMLFile);
	if (!file.open(QIODevice::WriteOnly))
		qCritical() << "Error: Cannot create output file" << processedOutputXMLFile << ": " << qPrintable(file.errorString());
	else {
		QXmlStreamWriter stream;
		stream.setDevice(&file);
		stream.setAutoFormatting(true);
		stream.writeStartDocument();
		stream.writeStartElement("layouts");
		for (QVector<QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, qreal>>::iterator it = solutionsCompilation.begin(); it != solutionsCompilation.end(); it++) {
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> curSolution = (*it).first;
			if (algorithmParamsBackup.getClusterFactor() > 0 && curSolution->getNumItems() > problem->count())
				curSolution->save(stream, std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(problem)->getOriginalProblem(), (*it).second, true, seed);
			else curSolution->save(stream, problem, (*it).second, true, seed);
		}
		stream.writeEndElement(); // layouts
		file.close();
	}
}

void ConsolePackingLoader::threadFinished() {
	numProcesses--;
	if (numProcesses == 0) emit quitApp();
}