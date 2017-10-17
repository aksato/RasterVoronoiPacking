#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

enum RasterPackingMethods {Method_Default, Method_Gls, Method_Zoom, Method_ZoomGls};
enum InitialSolutionGenerator {Solution_Random, Bottom_Left};
enum RectangularMethod { SQUARE, RANDOM_ENCLOSED, COST_EVALUATION, BAGPIPE };
enum ZoomMethod { Zoom_Rounded, Zoom_Distributed, Zoom_Weighted, Zoom_Single };

struct ConsolePackingArgs {
	ConsolePackingArgs() {}

    QString inputFilePath;
    QString zoomedInputFilePath;
    QString outputTXTFile;
    QString outputXMLFile;

    RasterPackingMethods methodType;
    InitialSolutionGenerator initialSolutionType;
	RectangularMethod rectMehod;
	ZoomMethod zoomMethod;
    int maxWorseSolutionsValue;
    int timeLimitValue;
	int iterationsLimitValue;
    qreal containerLenght;
	qreal clusterFactor;
	qreal rdec, rinc;

    bool originalContainerLenght;
	bool stripPacking;
	bool rectangularPacking;
	bool appendSeedToOutputFiles;

	int numThreads;
};

enum CommandLineParseResult
{
    CommandLineOk,
    CommandLineError,
    CommandLineVersionRequested,
    CommandLineHelpRequested
};

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, ConsolePackingArgs *params, QString *errorMessage);
CommandLineParseResult parseOptionsFile(QString fileName, ConsolePackingArgs *params, QString *errorMessage);

#endif // PACKINGPARAMETERSPARSER_H
