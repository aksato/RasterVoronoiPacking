#include <QCommandLineParser>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include "packingParametersParser.h"

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PackingBatchExecutorArgs *params, QString *errorMessage)
{
    parser.setSingleDashWordOptionMode(QCommandLineParser::ParseAsLongOptions);

	parser.addPositionalArgument("executable", "Path to Raster Packing Console executable.");
    parser.addPositionalArgument("source","Input problems file.");
	const QCommandLineOption valueNumThreads("parallel", "Number of parallel executions of the algorithm.", "value"); parser.addOption(valueNumThreads); 
	const QCommandLineOption valueNumExecutions("executions", "Number of total executions of the algorithm per case.", "value"); parser.addOption(valueNumExecutions);
	const QCommandLineOption valueCluster("clusterfactor", "Time fraction for cluster executuion.", "value"); parser.addOption(valueCluster);
	const QCommandLineOption valueRectangular("rectpacking", "Rectangular packing problem. Choices: square, random, bagpipe.", "value"); parser.addOption(valueRectangular);
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
    if (positionalArguments.isEmpty()) {
        *errorMessage = "Arguments missing: 'executable' and 'source'.";
        return CommandLineError;
    }
	if (positionalArguments.size() < 2) {
		*errorMessage = "Argument missing : 'source'.";
		return CommandLineError;
	}
    if (positionalArguments.size() > 2) {
        *errorMessage = "Too many arguments specified.";
        return CommandLineError;
    }
	
	params->executablePath = positionalArguments.at(0);
    params->inputFilePath = positionalArguments.at(1);

	bool parseOk;
	if (parser.isSet(valueNumThreads)) {
		const int nthreads = parser.value(valueNumThreads).toInt(&parseOk);
		if (parseOk && nthreads > 0) params->threadCount = nthreads;
		else { *errorMessage = "Bad parallel value."; return CommandLineError; }
	}
	if (parser.isSet(valueNumExecutions)) {
		const int nexecs = parser.value(valueNumExecutions).toInt(&parseOk);
		if (parseOk && nexecs > 0) params->executionCount = nexecs;
		else { *errorMessage = "Bad executions count value."; return CommandLineError; }
	}
	if (parser.isSet(valueCluster)) {
		const float clusterFactor = parser.value(valueCluster).toFloat(&parseOk);
		if (parseOk && clusterFactor >= 0 && clusterFactor <= 1.0) params->clusterFactor = clusterFactor;
		else { *errorMessage = "Bad cluster factor value."; return CommandLineError; }
	}

	if (parser.isSet(valueRectangular)) {
		params->rectangular = true;
		const QString methodType = parser.value(valueRectangular).toLower();
		if (methodType != "square" && methodType != "random" && methodType != "bagpipe") {
			*errorMessage = "Invalid initial rectangular method type! Avaible methods: 'square', 'random', 'bagpipe'.";
			return CommandLineError;
		}
		params->rectMehod = methodType;
	}

    return CommandLineOk;
}
