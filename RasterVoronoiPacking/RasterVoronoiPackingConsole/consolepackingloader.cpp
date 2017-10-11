#include "consolepackingloader.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/packing2dthread.h"
#include "raster/packingclusterthread.h"
#include "packingproblem.h"
#include <QDir>
#include <QXmlStreamWriter>
#include <iostream>
#include <iomanip>

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

ConsolePackingLoader::ConsolePackingLoader(QObject *parent) {
	numProcesses = 0;
}

bool ConsolePackingLoader::loadInputFile(QString inputFilePath, std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> *problem) {
	RASTERPACKING::PackingProblem preProblem;
	if (!preProblem.load(inputFilePath)) {
		qCritical("Could not open file '%s'!", qPrintable(inputFilePath));
		return false;
	}
	// Density calculation
	totalArea = preProblem.getTotalItemsArea();
	containerWidth = getContainerWidth(preProblem);

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
		//if (algorithmParamsBackup.getHeuristic() == RASTERVORONOIPACKING::NONE) solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(problem));
		//else {
		//	if (!algorithmParamsBackup.isDoubleResolution()) solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(problem));
		//	else solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(problem, zoomProblem));
		//}
		// Get container initial length
		int initialWidth = algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED ? qRound(algorithmParamsBackup.getInitialLenght()*problem->getScale()) : -1;
		solver = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver({ problem, zoomProblem }, algorithmParamsBackup, initialWidth);
		if (algorithmParamsBackup.isRectangularPacking()) {
			threadedPacker = std::shared_ptr<Packing2DThread>(new Packing2DThread);
			std::shared_ptr<Packing2DThread> threadedPacker2D = std::dynamic_pointer_cast<Packing2DThread>(threadedPacker);
			threadedPacker2D->setSolver(solver);
			threadedPacker2D->setMethod(algorithmParamsBackup.getRectangularPackingMethod());
		}
		else {
			threadedPacker = std::shared_ptr<PackingThread>(new PackingThread);
			threadedPacker->setSolver(solver);
		}
	}
	else {
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingClusterSolver> clusterSolverGls;
		std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> originalSolverGls;
		std::shared_ptr<RASTERVORONOIPACKING::RasterOverlapEvaluator> clusterOverlapEvaluator, originalOverlapEvaluator;
		// Get pointer to cluster problem
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(problem);
		if (algorithmParamsBackup.isDoubleResolution()) {
			// Get pointer to cluster zoomed problem
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterSearchProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(zoomProblem);
			//clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterDoubleGLS(clusterProblem, clusterSearchProblem));
			//originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverDoubleGLS(clusterProblem->getOriginalProblem(), clusterSearchProblem->getOriginalProblem()));
			clusterOverlapEvaluator = std::shared_ptr<RASTERVORONOIPACKING::RasterOverlapEvaluator>(new RASTERVORONOIPACKING::RasterOverlapEvaluatorDoubleGLS(clusterProblem, clusterSearchProblem));
			originalOverlapEvaluator = std::shared_ptr<RASTERVORONOIPACKING::RasterOverlapEvaluator>(new RASTERVORONOIPACKING::RasterOverlapEvaluatorDoubleGLS(clusterProblem->getOriginalProblem(), clusterSearchProblem->getOriginalProblem()));
		}
		else {
			//clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverClusterGLS(clusterProblem));
			//originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolverGLS>(new RASTERVORONOIPACKING::RasterStripPackingSolverGLS(clusterProblem->getOriginalProblem()));
			clusterOverlapEvaluator = std::shared_ptr<RASTERVORONOIPACKING::RasterOverlapEvaluator>(new RASTERVORONOIPACKING::RasterOverlapEvaluatorGLS(clusterProblem));
			originalOverlapEvaluator = std::shared_ptr<RASTERVORONOIPACKING::RasterOverlapEvaluator>(new RASTERVORONOIPACKING::RasterOverlapEvaluatorGLS(clusterProblem->getOriginalProblem()));
		}
		clusterSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingClusterSolver>(new RASTERVORONOIPACKING::RasterStripPackingClusterSolver(clusterProblem, clusterOverlapEvaluator));
		originalSolverGls = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(clusterProblem->getOriginalProblem(), originalOverlapEvaluator));
		// Configure Thread
		threadedPacker = std::shared_ptr<PackingClusterThread>(new PackingClusterThread);
		std::shared_ptr<PackingClusterThread> threadedClusterPacker = std::dynamic_pointer_cast<PackingClusterThread>(threadedPacker);
		threadedClusterPacker->setSolver(originalSolverGls, clusterSolverGls);
		connect(&*threadedClusterPacker, SIGNAL(unclustered(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)), this, SLOT(updateUnclusteredProblem(RASTERVORONOIPACKING::RasterPackingSolution, int, qreal)));
	}
	
	//// Resize container to initial length
	//RASTERVORONOIPACKING::RasterPackingSolution solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count());
	//qreal length;
	//if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) {
	//	length = algorithmParamsBackup.getInitialLenght();
	//	int initialWidth = qRound(length*problem->getScale());
	//	solver->setContainerWidth(initialWidth, solution, algorithmParamsBackup);
	//}

	// Configure packer object
	threadVector.push_back(threadedPacker);
	connect(&*threadedPacker, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(printExecutionStatus(int, int, int, qreal, qreal, qreal)));
	qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	qRegisterMetaType<ExecutionSolutionInfo>("ExecutionSolutionInfo");
	connect(&*threadedPacker, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)), SLOT(saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)));
	connect(&*threadedPacker, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)), SLOT(saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)));
	connect(&*threadedPacker, SIGNAL(finished()), SLOT(threadFinished()));
	threadedPacker->setParameters(algorithmParamsBackup);

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	if (!algorithmParamsBackup.isDoubleResolution()) qDebug() << "Problem Scale:" << problem->getScale();
	else qDebug() << "Problem Scale:" << zoomProblem->getScale() << ". Auxiliary problem scale:" << problem->getScale();
	if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) qDebug() << "Length:" << algorithmParamsBackup.getInitialLenght();
	qDebug() << "Solver method:" << algorithmParamsBackup.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParamsBackup.getInitialSolMethod();
	qDebug() << "Minimum overlap placement heuristic:" << algorithmParamsBackup.getPlacementCriteria();
	if (!algorithmParamsBackup.isFixedLength()) qDebug() << "Strip packing version";
	qDebug() << "Solver parameters: Nmo =" << algorithmParamsBackup.getNmo() << "; Time Limit:" << algorithmParamsBackup.getTimeLimit();
	if (algorithmParamsBackup.getClusterFactor() > 0) qDebug() << "Cluster factor:" << algorithmParamsBackup.getClusterFactor();
	if (algorithmParamsBackup.isRectangularPacking()) qDebug() << "Rectangular Packing Method: " << algorithmParamsBackup.getRectangularPackingMethod();
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
	return QDir(fileInfo.path()).filePath(newFileName);
}

void ConsolePackingLoader::writeNewLength(int length, int totalItNum, qreal elapsed, uint threadSeed) {
	QString *threadOutlogContens = outlogContents[threadSeed]; 
	if (!threadOutlogContens) { threadOutlogContens = new QString; outlogContents.insert(threadSeed, threadOutlogContens); }
	QTextStream out(threadOutlogContens);
	if (!algorithmParamsBackup.isDoubleResolution())
		out << problem->getScale() << " - " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else
		out << problem->getScale() << " " << zoomProblem->getScale() << " " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
}

void ConsolePackingLoader::writeNewLength2D(const ExecutionSolutionInfo &info, int totalItNum, qreal elapsed, uint threadSeed) {
	QString *threadOutlogContens = outlogContents[threadSeed];
	if (!threadOutlogContens) { threadOutlogContens = new QString; outlogContents.insert(threadSeed, threadOutlogContens); }
	QTextStream out(threadOutlogContens);
	if (!algorithmParamsBackup.isDoubleResolution())
		out << problem->getScale() << " - " << info.length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else
		out << problem->getScale() << " " << zoomProblem->getScale() << " " << info.length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
}

void ConsolePackingLoader::saveXMLSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	//qreal realLength = info.length / problem->getScale();
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> newSolution(new RASTERVORONOIPACKING::RasterPackingSolution(solution.getNumItems()));
	for (int i = 0; i < solution.getNumItems(); i++) {
		newSolution->setPosition(i, solution.getPosition(i));
		newSolution->setOrientation(i, solution.getOrientation(i));
	}
	solutionsCompilation.push_back(QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, ExecutionSolutionInfo>(newSolution, info));
}

void ConsolePackingLoader::saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	if (!info.twodim) {
		std::cout << "\n" << "New layout obtained: " << info.length / problem->getScale() << ". Elapsed time: " << info.timestamp << " secs. Seed = " << info.seed << "\n";
		writeNewLength(info.length, info.iteration, info.timestamp, info.seed);
	}
	else {
		std::cout << "\n" << "New layout obtained: " << info.length / problem->getScale() << "x" << info.height / problem->getScale() << " ( area = " << (info.length * info.height) / (problem->getScale() * problem->getScale()) << "). Elapsed time: " << info.timestamp << " secs. Seed = " << info.seed << "\n";
		writeNewLength2D(info, info.iteration, info.timestamp, info.seed);
	}
	saveXMLSolution(solution, info);
	solutionInfoHistory[info.seed].push_back(info);
}

void ConsolePackingLoader::updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed) {
	// Print to console
	std::cout << "\n" << "Undoing the initial clusters, returning to original problem. Current length:" << length / problem->getScale() << ". Elapsed time: " << elapsed << " secs" << "\n";
}

void ConsolePackingLoader::saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, const ExecutionSolutionInfo &info, int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime) {
	if (algorithmParamsBackup.isFixedLength()) 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << totalTime;
	else {
		if (!info.twodim) qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum length =" << info.length / problem->getScale() << ". Elapsed time:" << totalTime;
		else qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum area =" << info.area / (problem->getScale() *problem->getScale()) << ". Elapsed time:" << totalTime;
	}
	
	if(!info.twodim) writeNewLength(info.length, totalIt, totalTime, info.seed);
	else writeNewLength2D(info, totalIt, totalTime, info.seed);

	// Determine output file names
	uint seed = info.seed;
	QString processedOutputTXTFile = appendSeedToOutputFiles ? getSeedAppendedString(outputTXTFile, seed) : outputTXTFile;
	QString processedOutputXMLFile = appendSeedToOutputFiles ? getSeedAppendedString(outputXMLFile, seed) : outputXMLFile;

	// Print fixed container final result to file
	if (algorithmParamsBackup.isFixedLength()) {
		QFile file(processedOutputTXTFile);
		if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
		QTextStream out(&file);
		if (!algorithmParamsBackup.isDoubleResolution())
			out << problem->getScale() << " - " << info.length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		else
			out << problem->getScale() << " " << zoomProblem->getScale() << " " << info.length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		file.close();
	}
	else {
		QFile file(processedOutputTXTFile);
		if (!file.open(QIODevice::Append | QIODevice::Text)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
		QTextStream out(&file);
		out << *outlogContents[seed];
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
		for (QVector<QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, ExecutionSolutionInfo>>::iterator it = solutionsCompilation.begin(); it != solutionsCompilation.end(); it++) {
			std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> curSolution = (*it).first;
			qreal realLength = (*it).second.length / problem->getScale();
			if (algorithmParamsBackup.getClusterFactor() > 0 && curSolution->getNumItems() > problem->count())
				curSolution->save(stream, std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(problem)->getOriginalProblem(), realLength, true, seed);
			else {
				if (!info.twodim) curSolution->save(stream, problem, realLength, true, seed);
				else {
					qreal realHeight = (*it).second.height / problem->getScale();
					curSolution->save(stream, problem, realLength, realHeight, (*it).second.iteration, true, seed);
				}
			}
		}
		stream.writeEndElement(); // layouts
		file.close();
	}

	// Print compiled results
	auto bestResult = std::min_element(solutionInfoHistory[seed].begin(), solutionInfoHistory[seed].end(),
		[](ExecutionSolutionInfo const & lhs, ExecutionSolutionInfo const & rhs) {
		if (lhs.area == rhs.area) return lhs.iteration < rhs.iteration;
		return lhs.area < rhs.area;
	});
	QFileInfo fileInfo(processedOutputTXTFile);
	QString compilationFileName = QDir(fileInfo.path()).filePath("compiledResults.txt");
	QFile fileComp(compilationFileName);
	if (!fileComp.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
	else {
		QTextStream out(&fileComp);
		out << problem->getScale() << "\t" << (problem->getScale()*problem->getScale()*totalArea) / bestResult->area << "\t" << bestResult->length / problem->getScale() << "\t" <<
			(bestResult->twodim ? QString::number(bestResult->height / problem->getScale()) : "-") << "\t" << bestResult->area / (problem->getScale()*problem->getScale()) << "\t" << bestResult->timestamp << "\t" << bestResult->iteration << "\t" <<
			totalTime << "\t" << totalIt << "\t" << seed << "\n";
		fileComp.close();
	}
	
}

void ConsolePackingLoader::threadFinished() {
	numProcesses--;
	if (numProcesses == 0) emit quitApp();
}