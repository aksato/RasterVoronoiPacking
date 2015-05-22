#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

enum RasterPackingMethods {Method_Default, Method_Gls, Method_Zoom, Method_ZoomGls};
enum InitialSolutionGenerator {Solution_Random, Bottom_Left};
enum MultiplePlacementChoice {Pos_BottomLeft, Pos_Random, Pos_Limits, Pos_Contour};

struct ConsolePackingArgs {
	ConsolePackingArgs() {}

    QString inputFilePath;
    QString zoomedInputFilePath;
    QString outputTXTFile;
    QString outputXMLFile;

    RasterPackingMethods methodType;
    InitialSolutionGenerator initialSolutionType;
	MultiplePlacementChoice placementType;
    int maxWorseSolutionsValue;
    int timeLimitValue;
    qreal containerLenght;

    bool originalContainerLenght;
	bool stripPacking;
	bool gpuProcessing;

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
