#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include "parametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PreProcessorParameters *params, QString *errorMessage)
{
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);
    parser.addPositionalArgument("source","Input ESICUP problem file path.");
    parser.addPositionalArgument("destination","Destination puzzle file.");
    const QCommandLineOption helpOption = parser.addHelpOption();
    const QCommandLineOption versionOption = parser.addVersionOption();

    if (!parser.parse(QCoreApplication::arguments())) {
        *errorMessage = parser.errorText();
        return CommandLineError;
    }

    if (parser.isSet(versionOption))
        return CommandLineVersionRequested;

    if (parser.isSet(helpOption))
        return CommandLineHelpRequested;

    const QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.isEmpty() || positionalArguments.size() == 1) {
        *errorMessage = "Arguments missing: 'source' or 'destination' or both.";
        return CommandLineError;
    }
    if (positionalArguments.size() > 2) {
        *errorMessage = "Several arguments specified.";
        return CommandLineError;
    }
    params->inputFilePath = positionalArguments.at(0);
    params->outputFilePath = positionalArguments.at(1);

    return CommandLineOk;
}