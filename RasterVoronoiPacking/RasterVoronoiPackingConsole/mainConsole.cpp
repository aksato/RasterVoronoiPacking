#include <QCoreApplication>

#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/packingthread.h"
#include "consolepackingloader.h"
#include "args/args.hxx"
#include <QDebug>
#include <QDir>
#include <QCommandLineParser>
#include <QFile>
#include <QFileInfo>
#include <memory>
#include <iostream>

std::tuple<qreal, qreal> parseRatios(std::string values) {
	const QString ratiosStr = QString::fromStdString(values);
	QStringList ratiosList = ratiosStr.split(";");
	if(ratiosList.length() != 2) return { -1,-1 };
	bool ok;
	qreal rdec = ratiosList[0].toDouble(&ok);
	qreal rinc = ratiosList[1].toDouble(&ok);
	if (ok && rdec > 0 && rinc > 0) return { rdec,rinc };
	return { -1,-1 };
}

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName("Raster Packing Console");
	QCoreApplication::setApplicationVersion("0.1");

	// --> Parse command line arguments
	args::ArgumentParser parser("Raster Packing console version.", "No comments.");
	args::HelpFlag help(parser, "help", "Display this help menu", { 'h', "help" });
	args::CompletionFlag completion(parser, { "complete" });
	args::ValueFlag<std::string> argMethod(parser, "type", "Raster packing method choices: default, gls", { "method" });
	args::ValueFlag<std::string> argInitialSolution(parser, "type", "Initial solution choices: random, bottomleft", { "initial" });
	args::ValueFlag<int> argDuration(parser, "value", "Time limit in seconds", { "duration" });
	args::ValueFlag<int> argNmo(parser, "value", "Maximum number of non-best solutions", { "nmo" });
	args::ValueFlag<int> argMaxIterations(parser, "value", "Maximum number of iterations", { "maxits" });
	args::ValueFlag<int> argContainerLength(parser, "value", "Container lenght", { "length" });
	args::ValueFlag<int> argZoom(parser, "zoom", "Zoom value for double resolution search method", { "zoom" });
	args::Flag argDisableCache(parser, "flag", "Disable overlap map caching", { "disable-cache" });
	args::ValueFlag<int> argNumThreads(parser, "value", "Number of parallel executions of the algorithm", { "parallel" });
	args::ValueFlag<int> argParallelGroupSize(parser, "value", "Size of thread groups with shared data", { "parallel-group" });
	args::Flag argStrippacking(parser, "flag", "Strip packing version", { "strippacking" });
	args::ValueFlag<std::string> argRectangularpacking(parser, "type", "Rectangular packing version. Choices: square, random, bagpipe", { "rectpacking" });
	args::Flag argCuttingstock(parser, "flag", "Cutting stock version", { "cuttingstock" });
	args::ValueFlag<std::string> argSizingRatios(parser, "value;value", "Ratios for container increase / decrease given in rdec;rinc form", { "ratios" });
	args::ValueFlag<std::string> argTXToutfile(parser, "name", "The output result statistics file name", { "result" });
	args::ValueFlag<std::string> argXMLoutfile(parser, "name", "The output layout XML file name", { "layout" });
	args::Flag argSeedFlag(parser, "flag", "Automatically append seed value to output file names", { "appendseed" });
	args::Positional<std::string> argPuzzle(parser, "source", "Input problem file path");
	try { parser.ParseCLI(argc, argv); }
	catch (args::Completion e) { std::cout << e.what(); return 0; }
	catch (args::Help) { std::cout << parser; return 0; }
	catch (args::ParseError e) { std::cerr << e.what() << std::endl << parser; return 1; }

	// Transform source path to absolute
	QFileInfo inputFileInfo(QString::fromStdString(args::get(argPuzzle)));
	QString inputFilePath = inputFileInfo.absoluteFilePath();
	// Check if source paths exist
	if (!QFile(inputFilePath).exists()) {
		qCritical() << "Input file not found.";
		return 1;
	}

	// Configure parameters
	RASTERVORONOIPACKING::RasterStripPackingParameters algorithmParams;
	if(!argMethod) { qWarning() << "Warning: Method not specified, set to default (no zoom, gls)"; algorithmParams.setHeuristic(RASTERVORONOIPACKING::GLS); }
	else if(args::get(argMethod) == "default" ) algorithmParams.setHeuristic(RASTERVORONOIPACKING::NONE);
	else if (args::get(argMethod) == "gls") algorithmParams.setHeuristic(RASTERVORONOIPACKING::GLS);
	else { std::cerr << "Invalid method type! Avaible methods: 'default', 'gls'." << std::endl << parser; return 1;  }
	// Fixed length
	algorithmParams.setFixedLength(!argStrippacking && !argRectangularpacking);
	// Sizing ratios
	auto[rdec, rinc] = parseRatios(args::get(argSizingRatios)); if(argSizingRatios && (rdec < 0 || rinc < 0)) { std::cerr << "Bad ratio value." << std::endl << parser; return 1; }
	if (argSizingRatios) algorithmParams.setResizeChangeRatios(rdec, rinc);
	// Time limit
	if (argDuration) { if (args::get(argDuration) > 0) algorithmParams.setTimeLimit(args::get(argDuration)); else { std::cerr << "Bad time limit value (must be expressed in seconds)." << std::endl << parser; return 1; } }
	else { qWarning() << "Warning: Time limit not found, set to default (600s)."; algorithmParams.setTimeLimit(600); }
	// Number of worse solutions
	if (argNmo) { if (args::get(argNmo) > 0) algorithmParams.setNmo(args::get(argNmo)); else { std::cerr << "Bad Nmo value." << std::endl << parser; return 1; } }
	else { qWarning() << "Warning: Nmo not found, set to default (200)."; algorithmParams.setNmo(200); }
	// Iterations limit
	if (argMaxIterations) { if (args::get(argMaxIterations) > 0) algorithmParams.setIterationsLimit(args::get(argMaxIterations)); else { std::cerr << "Bad iteration limit value." << std::endl << parser; return 1; } }
	else algorithmParams.setIterationsLimit(0);
	// Initial solution	
	if (!argInitialSolution) { qWarning() << "Warning: Initial solution not specified, set to default (bottomleft)"; algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::BOTTOMLEFT); }
	else if (args::get(argInitialSolution) == "random") {
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::RANDOMFIXED);
		if(!argContainerLength || args::get(argContainerLength) < 0) { std::cerr << "Bad or inexisting container lenght value." << std::endl << parser; return 1; }
		algorithmParams.setInitialDimensions(args::get(argContainerLength));
	}
	else if (args::get(argInitialSolution) == "bottomleft"){ algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::BOTTOMLEFT); }
	else { std::cerr << "Invalid initial solution type! Avaible methods: 'random, bottomleft'." << std::endl << parser; return 1; }
	// Packing problem type
	if (!argRectangularpacking) {
		if(argCuttingstock) algorithmParams.setCompaction(RASTERVORONOIPACKING::CUTTINGSTOCK);
		else algorithmParams.setCompaction(RASTERVORONOIPACKING::STRIPPACKING);
	}
	else {
		if (args::get(argMethod) == "square") algorithmParams.setCompaction(RASTERVORONOIPACKING::SQUAREPACKING);
		else if (args::get(argMethod) == "random") algorithmParams.setCompaction(RASTERVORONOIPACKING::RECTRNDPACKING);
		else if (args::get(argMethod) == "bagpipe") algorithmParams.setCompaction(RASTERVORONOIPACKING::RECTBAGPIPEPACKING);
		else { std::cerr << "Invalid initial rectangular method type! Avaible methods: 'square', 'random', 'cost' and 'bagpipe'." << std::endl << parser; return 1; }
	}
	// Multiresolution zoom value
	if(!argZoom) algorithmParams.setZoomFactor(1); else if(args::get(argZoom) < 1) { std::cerr << "Bad zoom value." << std::endl << parser; return 1; }
	else algorithmParams.setZoomFactor(args::get(argZoom));
	// Cache maps
	algorithmParams.setCacheMaps(argDisableCache);
	// Thread number and group size
	int numThreads, threadGroupSize;
	if (!argNumThreads) numThreads = 1; else if (args::get(argNumThreads) < 1) { std::cerr << "Bad parallel value." << std::endl << parser; return 1; }
	else numThreads = args::get(argNumThreads);
	if (!argParallelGroupSize) threadGroupSize = 1; else if (args::get(argParallelGroupSize) < 1 || args::get(argParallelGroupSize) > argNumThreads) { std::cerr << "Bad thread group size value." << std::endl << parser; return 1; }
	else numThreads = args::get(argParallelGroupSize);
	// Output files


	int finishedThreadNums = 0;
	QVector<std::shared_ptr<ConsolePackingLoader>> packingLoadersList;
	std::shared_ptr<ConsolePackingLoader> currentPackingLoader;
	if (numThreads > 1) qDebug() << "Multithreaded version:" << numThreads << "parallel executions.";
	for (int i = 0; i < numThreads; i++) {
		if (i % threadGroupSize == 0) {
			currentPackingLoader = std::shared_ptr<ConsolePackingLoader>(new ConsolePackingLoader);
			QObject::connect(
				&*currentPackingLoader, &ConsolePackingLoader::quitApp,
				[&app, &finishedThreadNums, &numThreads, &threadGroupSize]() { finishedThreadNums += threadGroupSize; if (finishedThreadNums >= numThreads) app.quit(); }
			);
			currentPackingLoader->setParameters(inputFilePath, QString::fromStdString(argTXToutfile ? args::get(argTXToutfile) : "outlog.dat"), 
				QString::fromStdString(argXMLoutfile ? args::get(argXMLoutfile) : "bestSol.xml"), algorithmParams, argSeedFlag);
			packingLoadersList << currentPackingLoader;
		}
		currentPackingLoader->run();
		QThread::sleep(2);
	}
    return app.exec();
}
