#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

struct PackingBatchExecutorArgs {
	PackingBatchExecutorArgs() : executionCount(100), threadCount(1), clusterFactor(-1.0) {}
	QString executablePath;
    QString inputFilePath;
	int executionCount;
	int threadCount;
	qreal clusterFactor;
	bool rectangular;
	QString rectMehod;
};

enum CommandLineParseResult
{
    CommandLineOk,
    CommandLineError,
    CommandLineVersionRequested,
    CommandLineHelpRequested
};

CommandLineParseResult parseCommandLine(QCommandLineParser &parser, PackingBatchExecutorArgs *params, QString *errorMessage);

#endif // PACKINGPARAMETERSPARSER_H
