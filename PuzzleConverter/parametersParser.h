#ifndef PARSECOMMANDLINE_H
#define PARSECOMMANDLINE_H

class QCommandLineParser;
class QString;

struct PreProcessorParameters {
	PreProcessorParameters() {}

    QString inputFilePath;
    QString outputFilePath;
};

enum CommandLineParseResult
{
    CommandLineOk,
    CommandLineError,
    CommandLineVersionRequested,
    CommandLineHelpRequested
};

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PreProcessorParameters *params, QString *errorMessage);

#endif // PARSECOMMANDLINE_H
