#ifndef PARSECOMMANDLINE_H
#define PARSECOMMANDLINE_H

class QCommandLineParser;
class QString;

struct PreProcessorParameters {
	PreProcessorParameters() {}

    QString inputFilePath;
    QString inputFileType;
    qreal puzzleScaleFactor; // Nofit polygon precision
    QString outputDir;
    qreal rasterScaleFactor; // Rasterization precision
	qreal scaleFixFactor; // Correction scale factor for CFREFP problems
    bool outputImages;
	bool skipDt, skipOutput;
    QString outputXMLName;
    QString headerFile;
    QString optionsFile;
};

enum CommandLineParseResult
{
    CommandLineOk,
    CommandLineError,
    CommandLineVersionRequested,
    CommandLineHelpRequested
};

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PreProcessorParameters *params, QString *errorMessage);
CommandLineParseResult parseOptionsFile(QString fileName, PreProcessorParameters *params, QString *errorMessage);

#endif // PARSECOMMANDLINE_H
