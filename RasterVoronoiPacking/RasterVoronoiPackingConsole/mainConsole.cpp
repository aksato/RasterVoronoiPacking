#include <QCoreApplication>

#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/packingthread.h"
#include "packingParametersParser.h"
#include "consolepackingloader.h"
#include <QDebug>
#include <QDir>
#include <QCommandLineParser>
#include <QFile>
#include <QFileInfo>
#include <memory>

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName("Raster Packing Console");
	QCoreApplication::setApplicationVersion("0.1");

	// --> Parse command line arguments
	QCommandLineParser parser;
	parser.setApplicationDescription("Raster Packing console version.");
	ConsolePackingArgs params;
	RASTERVORONOIPACKING::RasterStripPackingParameters algorithmParams;
	QString errorMessage;
	switch (parseCommandLine(parser, &params, &errorMessage)) {
	case CommandLineOk:
		break;
	case CommandLineError:
		fputs(qPrintable(errorMessage), stderr);
		fputs("\n\n", stderr);
		fputs(qPrintable(parser.helpText()), stderr);
		return 1;
	case CommandLineVersionRequested:
		printf("%s %s\n", qPrintable(QCoreApplication::applicationName()),
			qPrintable(QCoreApplication::applicationVersion()));
		return 0;
	case CommandLineHelpRequested:
		parser.showHelp();
		Q_UNREACHABLE();
	}

	// Transform source path to absolute
	QFileInfo inputFileInfo(params.inputFilePath);
	params.inputFilePath = inputFileInfo.absoluteFilePath();
	// Check if source paths exist
	if (!QFile(params.inputFilePath).exists()) {
		qCritical() << "Input file not found.";
		return 1;
	}

	// Configure parameters
	switch (params.methodType) {
	case Method_Default: algorithmParams.setHeuristic(RASTERVORONOIPACKING::NONE); break;
	case Method_Gls: algorithmParams.setHeuristic(RASTERVORONOIPACKING::GLS); break;
	}
	algorithmParams.setFixedLength(!params.stripPacking);
	if (params.rdec > 0 && params.rinc > 0) algorithmParams.setResizeChangeRatios(params.rdec, params.rinc);
	algorithmParams.setTimeLimit(params.timeLimitValue); algorithmParams.setNmo(params.maxWorseSolutionsValue);
	algorithmParams.setIterationsLimit(params.iterationsLimitValue);
	if (params.initialSolutionType == Solution_Random) {
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::RANDOMFIXED);
		algorithmParams.setInitialDimensions(params.containerLenght);
	}
	else if (params.initialSolutionType == Bottom_Left){
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::BOTTOMLEFT);
	}
	if (!params.rectangularPacking) {
		if(params.cuttingStock) algorithmParams.setCompaction(RASTERVORONOIPACKING::CUTTINGSTOCK);
		else algorithmParams.setCompaction(RASTERVORONOIPACKING::STRIPPACKING);
	}
	else {
		switch (params.rectMehod) {
		case SQUARE: algorithmParams.setCompaction(RASTERVORONOIPACKING::SQUAREPACKING); break;
		case RANDOM_ENCLOSED: algorithmParams.setCompaction(RASTERVORONOIPACKING::RECTRNDPACKING); break;
		case BAGPIPE: algorithmParams.setCompaction(RASTERVORONOIPACKING::RECTBAGPIPEPACKING); break;
		}
	}
	algorithmParams.setZoomFactor(params.zoomValue);
	algorithmParams.setCacheMaps(params.cacheMaps);

	int finishedThreadNums = 0;
	QVector<std::shared_ptr<ConsolePackingLoader>> packingLoadersList;
	std::shared_ptr<ConsolePackingLoader> currentPackingLoader;
	if (params.numThreads > 1) qDebug() << "Multithreaded version:" << params.numThreads << "parallel executions.";
	for (int i = 0; i < params.numThreads; i++) {
		if (i % params.threadGroupSize == 0) {
			currentPackingLoader = std::shared_ptr<ConsolePackingLoader>(new ConsolePackingLoader);
			QObject::connect(
				&*currentPackingLoader, &ConsolePackingLoader::quitApp,
				[&app, &finishedThreadNums, &params]() { finishedThreadNums += params.threadGroupSize; if (finishedThreadNums >= params.numThreads) app.quit(); }
			);
			currentPackingLoader->setParameters(params.inputFilePath, params.outputTXTFile, params.outputXMLFile, algorithmParams, params.appendSeedToOutputFiles);
			packingLoadersList << currentPackingLoader;
		}
		currentPackingLoader->run();
		QThread::sleep(2);
	}
    return app.exec();
}
