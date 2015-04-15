#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

enum RasterPackingMethods {Method_Default, Method_Gls, Method_Zoom, Method_ZoomGls};
enum InitialSolutionGenerator {Solution_Random};

struct ConsolePackingArgs {
	ConsolePackingArgs() {}

    QString inputFilePath;
    QString zoomedInputFilePath;
    QString outputTXTFile;
    QString outputXMLFile;

    RasterPackingMethods methodType;
    InitialSolutionGenerator initialSolutionType;
    int maxWorseSolutionsValue;
    int timeLimitValue;
    qreal containerLenght;

    bool originalContainerLenght;
	bool stripPacking;
	bool gpuProcessing;
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
