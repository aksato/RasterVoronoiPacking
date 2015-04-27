#include <QCoreApplication>

#include "packingproblem.h"
#include "raster/rasterpackingproblem.h"
#include "raster/rasterpackingsolution.h"
#include "raster/rasterstrippackingsolver.h"
#include "raster/rasterstrippackingparameters.h"
#include "raster/packingthread.h"
#include "cuda/gpuinfo.h"
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

    // Transform source and destination paths to absolute
    QFileInfo inputFileInfo(params.inputFilePath);
    params.inputFilePath = inputFileInfo.absoluteFilePath();

    // Check if source and destination paths exist
    if(!QFile(params.inputFilePath).exists()) {
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
	algorithmParams.setGpuProcessing(params.gpuProcessing);
	algorithmParams.setTimeLimit(params.timeLimitValue); algorithmParams.setNmo(params.maxWorseSolutionsValue);
	if (params.initialSolutionType == Solution_Random) {
		algorithmParams.setInitialSolMethod(RASTERVORONOIPACKING::RANDOMFIXED);
		algorithmParams.settInitialLenght(params.containerLenght);
	}

	ConsolePackingLoader packingLoader;
	if (!algorithmParams.isDoubleResolution()) packingLoader.setParameters(params.inputFilePath, params.outputTXTFile, params.outputXMLFile, algorithmParams);
	else packingLoader.setParameters(params.inputFilePath, params.zoomedInputFilePath, params.outputTXTFile, params.outputXMLFile, algorithmParams);

	packingLoader.run();
 //   std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> problem, zoomProblem;
 //   RASTERVORONOIPACKING::RasterPackingSolution solution;
 //   std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
 //   RASTERPREPROCESSING::PackingProblem preProblem, preZoomProblem;
 //   PackingThread singleThreadedPacker;
	//ConsolePackingLoader packingLoader;
 //   qDebug() << "Program execution started.";

 //   // Load input file
 //   qDebug() << "Loading problem file...";
 //   qDebug() << "Input file:" << params.inputFilePath;
 //   QString originalPath = QDir::currentPath();
 //   QDir::setCurrent(QFileInfo(params.inputFilePath).absolutePath());
 //   if(!preProblem.load(params.inputFilePath)) {
 //       qCritical("Could not open file '%s'!", qPrintable(params.inputFilePath));
 //       return 1;
 //   }

	//// Check if it is possible to use GPU processing
	//if (params.gpuProcessing) {
	//	int numGPUs;  size_t freeCUDAMem, totalCUDAmem;
	//	if (CUDAPACKING::getTotalMemory(numGPUs, freeCUDAMem, totalCUDAmem)) {
	//		size_t problemIfpTotalMem, problemIfpMaxMem, problemNfpTotalMem;
	//		RASTERVORONOIPACKING::RasterPackingProblem::getProblemGPUMemRequirements(preProblem, problemIfpTotalMem, problemIfpMaxMem, problemNfpTotalMem);
	//		if (freeCUDAMem - problemIfpTotalMem - problemNfpTotalMem > 0) CUDAPACKING::allocDeviceMaxIfp(problemIfpMaxMem);
	//		else {
	//			qDebug() << "GPU check error: not enough memory";
	//			params.gpuProcessing = false;
	//		}
	//	}
	//	else {
	//		params.gpuProcessing = false;
	//		qDebug() << "GPU check error: could not get gpu memory information";
	//	}
	//}
	//if(params.gpuProcessing) qDebug() << "GPU check success";
	//// Create problem object
	//problem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
	//problem->load(preProblem, params.gpuProcessing);
 //   QDir::setCurrent(originalPath);
 //   // Create solver object
	//solution = RASTERVORONOIPACKING::RasterPackingSolution(problem->count(), params.gpuProcessing);
 //   solver = std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver>(new RASTERVORONOIPACKING::RasterStripPackingSolver(problem));
 //   qDebug() << "Problem file read successfully";

 //   // Load zoom input file
 //   if(params.methodType == Method_Zoom || params.methodType == Method_ZoomGls) {
 //       qDebug() << "Loading zoomed problem file...";
 //        qDebug() << "Zoom input file:" << params.zoomedInputFilePath;
 //        QDir::setCurrent(QFileInfo(params.zoomedInputFilePath).absolutePath());
 //        if(!preZoomProblem.load(params.zoomedInputFilePath)) {
 //            qCritical("Could not open file '%s'!", qPrintable(params.zoomedInputFilePath));
 //            return 1;
 //        }
 //        zoomProblem = std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem>(new RASTERVORONOIPACKING::RasterPackingProblem);
 //        zoomProblem->load(preZoomProblem);
 //        solver->setProblem(zoomProblem, true);
 //        QDir::setCurrent(originalPath);
 //        qDebug() << "Zoomed problem file read successfully";
 //   }

 //   qreal length;
 //   if(!params.originalContainerLenght) {
 //       length = params.containerLenght;
 //       int scaledWidth = qRound(length*problem->getScale());
 //       solver->setContainerWidth(scaledWidth);
 //   }
 //   else length = solver->getCurrentWidth()/problem->getScale();
	//if (params.initialSolutionType == Solution_Random)  solver->generateRandomSolution(solution, algorithmParams);
 //   singleThreadedPacker.setInitialSolution(solution);
	//singleThreadedPacker.setParameters(algorithmParams);
 //   singleThreadedPacker.setSolver(solver);
 //   if(params.methodType == Method_Default || params.methodType == Method_Gls) singleThreadedPacker.setScale(problem->getScale());
 //   if(params.methodType == Method_Zoom || params.methodType == Method_ZoomGls) singleThreadedPacker.setScale(zoomProblem->getScale(), problem->getScale());
 //   qDebug() << "Solver configured. The following parameters were set:";
 //   if(params.methodType == Method_Zoom || params.methodType == Method_ZoomGls)
 //       qDebug() << "Zoomed problem Scale:" << zoomProblem->getScale() << ". Auxiliary problem scale:" << problem->getScale();
 //   else qDebug() << "Problem Scale:" << problem->getScale();
 //   qDebug() << "Length:" << length;
 //   qDebug() << "Solver method:" << params.methodType;
 //   qDebug() << "Inital solution:" << params.initialSolutionType;
	//if(params.stripPacking) qDebug() << "Strip packing version";
	//if(params.gpuProcessing) qDebug() << "Using GPU to process maps";
 //   qDebug() << "Solver parameters: Nmo =" << params.maxWorseSolutionsValue << "; Time Limit:" << params.timeLimitValue;

 //   // Run!
 //   int totalIt; qreal minOverlap, totalTime;
 //   uint seed;
 //   singleThreadedPacker.run(length, minOverlap, totalTime, totalIt, seed, solution);
 //   if(params.methodType == Method_Default || params.methodType == Method_Gls) solution.save(params.outputXMLFile, problem, length, true, seed);
 //   if(params.methodType == Method_Zoom || params.methodType == Method_ZoomGls) solution.save(params.outputXMLFile, zoomProblem, length, true, seed);

 //   QFile file(params.outputTXTFile);
 //   if(!file.open(QIODevice::Append)) qCritical() << "Error: Cannot create output file" << params.outputTXTFile << ": " << qPrintable(file.errorString());
 //   QTextStream out(&file);
 //   if(params.methodType == Method_Zoom || params.methodType == Method_ZoomGls)
	//	out << problem->getScale() << " " << zoomProblem->getScale() << " " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt/totalTime << " " << seed << "\n";
 //   else
	//	out << problem->getScale() << " - " << length << " " << minOverlap << " " << totalIt << " " << totalTime << " " << totalIt/totalTime << " " << seed << "\n";
 //   file.close();
    return app.exec();
}
