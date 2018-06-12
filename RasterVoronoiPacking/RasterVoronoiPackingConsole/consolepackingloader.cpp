#include "consolepackingloader.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rastersquarepackingcompactor.h"
#include "raster/rasterrectpackingcompactor.h"
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
	*problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
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

void ConsolePackingLoader::run() {
	std::shared_ptr<PackingThread> threadedPacker;

	// Create solver object
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver = RASTERVORONOIPACKING::RasterStripPackingSolver::createRasterPackingSolver(problem, algorithmParamsBackup);
	threadedPacker = std::shared_ptr<PackingThread>(new PackingThread);

	// Create compactor
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor> compactor;
	switch (algorithmParamsBackup.getCompaction()) {
	case RASTERVORONOIPACKING::STRIPPACKING:
	case RASTERVORONOIPACKING::CUTTINGSTOCK: // FIXME: Implement cutting stock compactor
		compactor = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor>(new RASTERVORONOIPACKING::RasterStripPackingCompactor(algorithmParamsBackup.getInitialLength(), 
			problem, solver->getOverlapEvaluator(), algorithmParamsBackup.getRdec(), algorithmParamsBackup.getRinc())); 
		break;
	case RASTERVORONOIPACKING::SQUAREPACKING:
		compactor = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor>(new RASTERVORONOIPACKING::RasterSquarePackingCompactor(algorithmParamsBackup.getInitialLength(), 
			problem, solver->getOverlapEvaluator(), algorithmParamsBackup.getRdec(), algorithmParamsBackup.getRinc())); 
		break;
	case RASTERVORONOIPACKING::RECTRNDPACKING:
		compactor = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor>(new RASTERVORONOIPACKING::RasterRectangularPackingRandomCompactor(algorithmParamsBackup.getInitialLength(), algorithmParamsBackup.getInitialHeight(), 
			problem, solver->getOverlapEvaluator(), algorithmParamsBackup.getRdec(), algorithmParamsBackup.getRinc())); 
		break;
	case RASTERVORONOIPACKING::RECTBAGPIPEPACKING:
		compactor = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingCompactor>(new RASTERVORONOIPACKING::RasterRectangularPackingBagpipeCompactor(algorithmParamsBackup.getInitialLength(), algorithmParamsBackup.getInitialHeight(), 
			problem, solver->getOverlapEvaluator(), algorithmParamsBackup.getRdec(), algorithmParamsBackup.getRinc())); 
		break;
	}

	// Configure packer object
	threadVector.push_back(threadedPacker);
	connect(&*threadedPacker, SIGNAL(statusUpdated(int, int, int, qreal, qreal, qreal)), this, SLOT(printExecutionStatus(int, int, int, qreal, qreal, qreal)));
	qRegisterMetaType<RASTERVORONOIPACKING::RasterPackingSolution>("RASTERVORONOIPACKING::RasterPackingSolution");
	qRegisterMetaType<ExecutionSolutionInfo>("ExecutionSolutionInfo");
	connect(&*threadedPacker, SIGNAL(minimumLenghtUpdated(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)), SLOT(saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo)));
	connect(&*threadedPacker, SIGNAL(finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)), SLOT(saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution, const ExecutionSolutionInfo, int, qreal, qreal, qreal)));
	connect(&*threadedPacker, SIGNAL(finished()), SLOT(threadFinished()));
	threadedPacker->setSolver(solver);
	threadedPacker->setCompactor(compactor);
	threadedPacker->setParameters(algorithmParamsBackup);

	// Print configurations
	qDebug() << "Solver configured. The following parameters were set:";
	if (algorithmParamsBackup.getZoomFactor() > 1)
		qDebug() << "Problem Scale:" << problem->getScale() << ". Auxiliary problem scale:" << problem->getScale() / algorithmParamsBackup.getZoomFactor();
	else qDebug() << "Problem Scale:" << problem->getScale();
	if (algorithmParamsBackup.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) qDebug() << "Length:" << algorithmParamsBackup.getInitialLength();
	qDebug() << "Solver method:" << algorithmParamsBackup.getHeuristic();
	qDebug() << "Inital solution:" << algorithmParamsBackup.getInitialSolMethod();
	if (!algorithmParamsBackup.isFixedLength()) qDebug() << "Strip packing version";
	qDebug() << "Solver parameters: Nmo =" << algorithmParamsBackup.getNmo() << "; Time Limit:" << algorithmParamsBackup.getTimeLimit();
	qDebug() << "Packing Compaction Method: " << algorithmParamsBackup.getCompaction();
	if (algorithmParamsBackup.isCacheMaps()) qDebug() << "Caching maps";
	numProcesses++;
	// Run!
	threadedPacker->start();
}

void ConsolePackingLoader::printExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed) {
	if (algorithmParamsBackup.getZoomFactor() > 1)
		std::cout << std::fixed << std::setprecision(2) << "\r" << "L: " << curLength / problem->getScale() <<
		". It: " << totalItNum << " (" << worseSolutionsCount << "). Min overlap: " << minOverlap / (qreal)(problem->getScale() / algorithmParamsBackup.getZoomFactor()) << ". Time: " << elapsed << " s.";
	else
		std::cout << std::fixed << std::setprecision(2) << "\r" << "L: " << curLength / problem->getScale() <<
		". It: " << totalItNum << " (" << worseSolutionsCount << "). Min overlap: " << minOverlap / problem->getScale() << ". Time: " << elapsed << " s."; 
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
	if (algorithmParamsBackup.getZoomFactor() > 1)
		out << problem->getScale() << " " << problem->getScale() / algorithmParamsBackup.getZoomFactor() << " " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else 
		out << problem->getScale() << " - " << length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
}

void ConsolePackingLoader::writeNewLength2D(const ExecutionSolutionInfo &info, int totalItNum, qreal elapsed, uint threadSeed) {
	QString *threadOutlogContens = outlogContents[threadSeed];
	if (!threadOutlogContens) { threadOutlogContens = new QString; outlogContents.insert(threadSeed, threadOutlogContens); }
	QTextStream out(threadOutlogContens);
	if (algorithmParamsBackup.getZoomFactor() > 1)
		out << problem->getScale() << " " << problem->getScale() / algorithmParamsBackup.getZoomFactor() << " " << info.length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
	else
		out << problem->getScale() << " - " << info.length / problem->getScale() << " " << totalItNum << " " << elapsed << " " << totalItNum / elapsed << " " << threadSeed << "\n";
}

void ConsolePackingLoader::saveXMLSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	//qreal realLength = info.length / problem->getScale();
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> newSolution(new RASTERVORONOIPACKING::RasterPackingSolution(solution.getNumItems()));
	for (int i = 0; i < solution.getNumItems(); i++) {
		newSolution->setPosition(i, solution.getPosition(i));
		newSolution->setOrientation(i, solution.getOrientation(i));
	}
	// Add new solution
	solutionsCompilation[info.seed].push_back(QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, ExecutionSolutionInfo>(newSolution, info));
}

void ConsolePackingLoader::saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const ExecutionSolutionInfo &info) {
	ExecutionSolutionInfo infoCopy = info;
	if (info.pType == RASTERVORONOIPACKING::ProblemType::StripPacking) {
		//infoCopy.density = this->problem->getDensity((RASTERVORONOIPACKING::RasterPackingSolution&)solution);
		std::cout << "New layout obtained: " << info.length / problem->getScale() << ". Elapsed time: " << info.timestamp << " secs. Seed = " << info.seed << "\n";
		writeNewLength(info.length, info.iteration, info.timestamp, info.seed);
	}
	else { // RASTERVORONOIPACKING::ProblemType::SquarePacking || RASTERVORONOIPACKING::ProblemType::RectangularPacking
		//infoCopy.density = this->problem->getRectangularDensity((RASTERVORONOIPACKING::RasterPackingSolution&)solution);
		std::cout << "New layout obtained: " << info.length / problem->getScale() << "x" << info.height / problem->getScale() << " ( area = " << (info.length * info.height) / (problem->getScale() * problem->getScale()) << "). Elapsed time: " << info.timestamp << " secs. Seed = " << info.seed << "\n";
		writeNewLength2D(info, info.iteration, info.timestamp, info.seed);
	}
	solutionInfoHistory[info.seed].push_back(infoCopy);
	saveXMLSolution(solution, infoCopy);
}

void ConsolePackingLoader::updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed) {
	// Print to console
	std::cout << "\n" << "Undoing the initial clusters, returning to original problem. Current length:" << length / problem->getScale() << ". Elapsed time: " << elapsed << " secs" << "\n";
}

void ConsolePackingLoader::saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, const ExecutionSolutionInfo &info, int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime) {
	if (algorithmParamsBackup.isFixedLength()) 
		qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << totalTime;
	else {
		if (info.pType == RASTERVORONOIPACKING::ProblemType::StripPacking) qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum length =" << info.length / problem->getScale() << ". Elapsed time:" << totalTime;
		else qDebug() << "\nFinished. Total iterations:" << totalIt << ".Minimum area =" << info.area / (problem->getScale() *problem->getScale()) << ". Elapsed time:" << totalTime; // RASTERVORONOIPACKING::ProblemType::SquarePacking || RASTERVORONOIPACKING::ProblemType::RectangularPacking
	}
	
	if (info.pType == RASTERVORONOIPACKING::ProblemType::StripPacking) writeNewLength(info.length, totalIt, totalTime, info.seed);
	else writeNewLength2D(info, totalIt, totalTime, info.seed); // RASTERVORONOIPACKING::ProblemType::SquarePacking || RASTERVORONOIPACKING::ProblemType::RectangularPacking

	// Determine output file names
	uint seed = info.seed;
	QString processedOutputTXTFile = appendSeedToOutputFiles ? getSeedAppendedString(outputTXTFile, seed) : outputTXTFile;
	QString processedOutputXMLFile = appendSeedToOutputFiles ? getSeedAppendedString(outputXMLFile, seed) : outputXMLFile;

	// Print fixed container final result to file
	if (algorithmParamsBackup.isFixedLength()) {
		QFile file(processedOutputTXTFile);
		if (!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
		QTextStream out(&file);
		if (algorithmParamsBackup.getZoomFactor() > 1)
			out << problem->getScale() << " " << problem->getScale() / algorithmParamsBackup.getZoomFactor() << " " << info.length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
		else
			out << problem->getScale() << " - " << info.length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt / totalTime << " " << seed << "\n";
			
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
	for (auto &entry : solutionsCompilation[seed])
		switch (entry.second.pType) {
			case RASTERVORONOIPACKING::ProblemType::StripPacking:
				entry.second.density = this->problem->getDensity(*entry.first);
				break;
			case RASTERVORONOIPACKING::ProblemType::SquarePacking:
				entry.second.density = this->problem->getSquareDensity(*entry.first);
				break;
			case RASTERVORONOIPACKING::ProblemType::RectangularPacking:
				entry.second.density = this->problem->getRectangularDensity(*entry.first);
				break;
		}
	auto bestResult = std::min_element(solutionsCompilation[seed].begin(), solutionsCompilation[seed].end(),
		[](QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, ExecutionSolutionInfo> &lhs, QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, ExecutionSolutionInfo> & rhs) {
		//if (lhs.area == rhs.area) return lhs.iteration < rhs.iteration;
		return lhs.second.density > rhs.second.density;
	});

	QString processedOutputPGFFile = QFileInfo(processedOutputXMLFile).path() + QDir::separator() + QFileInfo(processedOutputXMLFile).baseName() + ".pgf";
	QFile file(processedOutputXMLFile);
	if (!file.open(QIODevice::WriteOnly))
		qCritical() << "Error: Cannot create output file" << processedOutputXMLFile << ": " << qPrintable(file.errorString());
	else {
		QXmlStreamWriter stream;
		stream.setDevice(&file);
		stream.setAutoFormatting(true);
		stream.writeStartDocument();
		stream.writeStartElement("layouts");
		//for (auto it = solutionsCompilation.begin(); it != solutionsCompilation.end(); it++) {
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution> curSolution = (*bestResult).first;
		qreal realLength = (*bestResult).second.length / problem->getScale();
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> solutionProblem = this->problem;
		if (info.pType == RASTERVORONOIPACKING::ProblemType::StripPacking) {
			curSolution->save(stream, solutionProblem, realLength, true, seed);
			curSolution->exportToPgf(processedOutputPGFFile, solutionProblem, realLength, solutionProblem->getOriginalHeight());
		}
		else { // RASTERVORONOIPACKING::ProblemType::SquarePacking || RASTERVORONOIPACKING::ProblemType::RectangularPacking
			qreal realHeight = (*bestResult).second.height / problem->getScale();
			curSolution->save(stream, solutionProblem, realLength, realHeight, (*bestResult).second.iteration, true, seed);
			curSolution->exportToPgf(processedOutputPGFFile, solutionProblem, realLength, realHeight);
		}
		//}
		stream.writeEndElement(); // layouts
		file.close();
	}

	// Print compiled results
	QFileInfo fileInfo(processedOutputTXTFile);
	QString compilationFileName = QDir(fileInfo.path()).filePath("compiledResults.txt");
	QFile fileComp(compilationFileName);
	if (!fileComp.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << processedOutputTXTFile << ": " << qPrintable(file.errorString());
	else {
		QTextStream out(&fileComp);
		out << problem->getScale() << "\t" << bestResult->second.density << "\t" << bestResult->second.length / problem->getScale() << "\t" << minOverlap << "\t" <<
			((bestResult->second.pType == RASTERVORONOIPACKING::ProblemType::SquarePacking || bestResult->second.pType ==  RASTERVORONOIPACKING::ProblemType::RectangularPacking) ? QString::number(bestResult->second.height / problem->getScale()) : "-") << 
			"\t" << bestResult->second.area / (problem->getScale()*problem->getScale()) << "\t" << bestResult->second.timestamp << "\t" << bestResult->second.iteration << "\t" <<
			totalTime << "\t" << totalIt << "\t" << seed << "\n";
		fileComp.close();
	}
	
}

void ConsolePackingLoader::threadFinished() {
	numProcesses--;
	if (numProcesses == 0) emit quitApp();
}