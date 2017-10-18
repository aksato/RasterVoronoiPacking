#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

struct PackingBatchExecutorArgs {
	PackingBatchExecutorArgs() : executionCount(100), threadCount(1), clusterFactor(-1.0), rdec(0.1), rinc(0.01), zoomValue(-1), rectangular(false) {}
	QString executablePath;
    QString inputFilePath;
	int executionCount;
	int threadCount;
	qreal clusterFactor;
	qreal rdec;
	qreal rinc;
	bool rectangular;
	QString rectMehod;
	int zoomValue;
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
