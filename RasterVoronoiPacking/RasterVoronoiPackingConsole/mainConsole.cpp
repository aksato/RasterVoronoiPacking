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
#include <windows.h>

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName("Raster Packing Console");
	QCoreApplication::setApplicationVersion("0.1");
	ConsolePackingLoader packingLoader;
	QObject::connect(&packingLoader, SIGNAL(quitApp()), &app, SLOT(quit()));

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

	// Transform source and destination paths to absolute
	QFileInfo inputFileInfo(params.inputFilePath);
	params.inputFilePath = inputFileInfo.absoluteFilePath();

	// Check if source and destination paths exist
	if (!QFile(params.inputFilePath).exists()) {
		qCritical() << "Input file not found.";
		return 1;
	}

	// Configure parameters
	switch (params.methodType) {
	case Method_Default: algorithmParams.setHeuristic(RASTERVORONOIPACKING::NONE); algorithmParams.setDoubleResolution(false); break;
	case Method_Gls: algorithmParams.setHeuristic(RASTERVORONOIPACKING::GLS); algorithmParams.setDoubleResolution(false); break;
	case Method_Zoom: algorithmParams.setHeuristic(RASTERVORONOIPACKING::NONE); algorithmParams.setDoubleResolution(true); break;
	case Method_ZoomGls: algorithmParams.setHeuristic(RASTERVORONOIPACKING::GLS); algorithmParams.setDoubleResolution(true); break;
	}
	algorithmParams.setFixedLength(!params.stripPacking);
	algorithmParams.setTimeLimit(params.timeLimitValue); algorithmParams.setNmo(params.maxWorseSolutionsValue);
	algorithmParams.setIterationsLimit(params.iterationsLimitValue);
	if (params.initialSolutionType == Solution_Random) {
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::RANDOMFIXED);
		algorithmParams.setInitialLenght(params.containerLenght);
	}
	else if (params.initialSolutionType == Bottom_Left){
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::BOTTOMLEFT);
	}
	switch (params.placementType) {
		case Pos_BottomLeft: algorithmParams.setPlacementCriteria(RASTERVORONOIPACKING::BOTTOMLEFT_POS); break;
		case Pos_Random: algorithmParams.setPlacementCriteria(RASTERVORONOIPACKING::RANDOM_POS); break;
		case Pos_Limits: algorithmParams.setPlacementCriteria(RASTERVORONOIPACKING::LIMITS_POS); break;
		case Pos_Contour: algorithmParams.setPlacementCriteria(RASTERVORONOIPACKING::CONTOUR_POS); break;
	}

	if (!algorithmParams.isDoubleResolution()) packingLoader.setParameters(params.inputFilePath, params.outputTXTFile, params.outputXMLFile, algorithmParams);
	else packingLoader.setParameters(params.inputFilePath, params.zoomedInputFilePath, params.outputTXTFile, params.outputXMLFile, algorithmParams);

	if (params.numThreads > 1) qDebug() << "Multithreaded version:" << params.numThreads << "parallel executions.";
	for (int i = 0; i < params.numThreads; i++) {
		//PackingThread threadedPacker;
		packingLoader.run();
		Sleep(10000);
	}
    return app.exec();
}
