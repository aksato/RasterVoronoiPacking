#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include "parametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PreProcessorParameters *params, QString *errorMessage)
{
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);
    parser.addPositionalArgument("source","Input ESICUP problem file path.");
    parser.addPositionalArgument("destination","Destination puzzle file.");
	const QCommandLineOption boolTerashima("terashima", "Terashima puzzle input problem."); parser.addOption(boolTerashima);
	const QCommandLineOption valueSubproblem("subproblem", "Generate puzzle for a subset of the items (for Terashima problems only).", "index"); parser.addOption(valueSubproblem);
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
	if (parser.isSet(valueSubproblem)) params->subProblem = parser.value(valueSubproblem).toInt(); else params->subProblem = -1;
	params->terashima = parser.isSet(boolTerashima);

    return CommandLineOk;
}