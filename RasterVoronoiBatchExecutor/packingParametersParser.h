#ifndef PACKINGPARAMETERSPARSER_H
#define PACKINGPARAMETERSPARSER_H

class QCommandLineParser;
class QString;

struct PackingBatchExecutorArgs {
	PackingBatchExecutorArgs() : executionCount(100), threadCount(1), clusterFactor(-1.0), rdec(0.04), rinc(0.01), rectangular(false), threadGroupSize(1), fixedLength(-1) {}
	QString executablePath;
    QString inputFilePath;
	int executionCount;
	int threadCount;
	qreal clusterFactor;
	qreal rdec;
	qreal rinc;
	bool rectangular;
	QString rectMehod;
	QString appendResultPath;
	int threadGroupSize;
	bool cacheMaps, cuttingstock;
	qreal fixedLength;
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
