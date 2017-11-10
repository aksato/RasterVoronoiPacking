#include <QCoreApplication>
#include <QCommandLineParser>
#include <QProcess>
#include <QDebug>
#include <QFile>
#include <QRegExp>
#include <QDir>
#include "packingParametersParser.h"

struct CaseExecutionParam {
	QString problemFileName;
	int timeLimit;
	QString outputFolder;
	int zoomFactor;
};

QStringList readCasesFile(QString fileName, QList<int> &numParameters) {
	QFile myTextFile(fileName);
	QStringList myStringList;

	if (!myTextFile.open(QIODevice::ReadOnly) || myTextFile.atEnd()) {
		qWarning() << "Error opening cases file:" << myTextFile.errorString();
	}
	else {
		while (!myTextFile.atEnd()) {
			QString curLine = myTextFile.readLine();
			QStringList curSplitLine = curLine.split(QRegExp("\\s+"), QString::SkipEmptyParts);
			myStringList << curSplitLine;
			numParameters.push_back(curSplitLine.length());
			if (curSplitLine.length() < 3 || curSplitLine.length() > 4) {
				qWarning() << "Error reading parameters from case file.";
				return QStringList();
			}
		}
		myTextFile.close();
	}
	return myStringList;
}

QList<CaseExecutionParam> parseCasesFile(QString fileName, QString *errorMessage) {
	QList<int> lineParameters;
	QStringList caseFileStringList = readCasesFile(fileName, lineParameters);
	if (caseFileStringList.empty()) {
		*errorMessage = "Error reading entries in cases file.";
		return QList<CaseExecutionParam>();
	}

	QList<CaseExecutionParam> caseList;
	int paramNum = 0;
	bool ok;
	for (int caseNum = 0; caseNum < lineParameters.length(); caseNum++) {
		int numParametersPerEntry = lineParameters[caseNum];
		CaseExecutionParam curCase;
		curCase.problemFileName = caseFileStringList[paramNum++];
		if (numParametersPerEntry == 4) {
			curCase.zoomFactor = caseFileStringList[paramNum++].toInt(&ok);
			if (!ok) { *errorMessage = "Could not parse zoom factor for case " + QString::number(caseNum + 1) + "."; return QList<CaseExecutionParam>(); }

		}
		else curCase.zoomFactor = 1;
		curCase.timeLimit = caseFileStringList[paramNum++].toInt(&ok);
		if (!ok) {
			*errorMessage = "Could not parse time limit for case " + QString::number(caseNum+1) + ".";
			return QList<CaseExecutionParam>();
		}
		curCase.outputFolder = caseFileStringList[paramNum++];
		caseList.append(curCase);
	}

	return caseList;
}

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName("Raster Packing Batch Executor");
	QCoreApplication::setApplicationVersion("0.1");

	// --> Parse command line arguments
	QCommandLineParser parser;
	parser.setApplicationDescription("Raster Packing console version.");
	PackingBatchExecutorArgs params;
	QString errorMessage;
	switch (parseCommandLine(parser, &params, &errorMessage)) {
	case CommandLineOk:
		break;
	case CommandLineError:
		fputs(qPrintable(errorMessage), stderr);
		fputs("\n\n", stderr);
		fputs(qPrintable(parser.helpText()), stderr);
		return 1;
	case CommandLineVersionRequested:
		printf("%s %s\n", qPrintable(QCoreApplication::applicationName()),
			qPrintable(QCoreApplication::applicationVersion()));
		return 0;
	case CommandLineHelpRequested:
		parser.showHelp();
		Q_UNREACHABLE();
	}

	// Parse case file
	QList<CaseExecutionParam> cases = parseCasesFile(params.inputFilePath, &errorMessage);
	if (cases.empty()) {
		fputs(qPrintable(errorMessage), stderr);
		fputs("\n\n", stderr);
		fputs(qPrintable(parser.helpText()), stderr);
		return 1;
	}

	
	QProcess *myProcess = new QProcess;
	myProcess->setProcessChannelMode(QProcess::MergedChannels);
	foreach (CaseExecutionParam curCase , cases)
	{
		// Determine output files from folder
		QDir outputDir(curCase.outputFolder + QDir::separator() + params.appendResultPath);
		QString xmlOutput = outputDir.filePath("bestSol.xml");
		QString txtOutput = outputDir.filePath("outlog.dat");

		// Count the number of outlog files
		int pastExecutionCount = outputDir.entryList(QStringList("outlog*.dat")).length();
		if (pastExecutionCount > 0) qDebug() << "Detected" << pastExecutionCount << "executions of case" << curCase.problemFileName << ". Resuming...";

		// Create argument list
		QStringList arguments;
		arguments << curCase.problemFileName;
		arguments << "--method=gls";
		if (curCase.zoomFactor > 1) arguments << "--zoom=" + QString::number(curCase.zoomFactor);
		arguments << "--initial=bottomleft" << "--duration=" + QString::number(curCase.timeLimit) << "--strippacking" << "--appendseed" << "--layout=" + xmlOutput << "--result=" + txtOutput;// << "--parallel=" + QString::number(params.threadCount);
		if (params.clusterFactor >= 0) arguments << "--clusterfactor=" + QString::number(params.clusterFactor); // FIXME: customize cluster factor for each execution
		if (params.rectangular) arguments << "--rectpacking=" + params.rectMehod;
		arguments << "--ratios=" + QString::number(params.rdec) + ";"  + QString::number(params.rinc);

		if (pastExecutionCount >= params.executionCount) { qDebug() << "Skipping execution of fully processed case" << curCase.problemFileName; continue; }		
		qDebug() << "Running Raster Packing Console with arguments" << arguments;
		for (int i = pastExecutionCount; i < params.executionCount; i += params.threadCount) {
			// Check the number of necessary threads to reach desired execution count
			int executionsCount = i + params.threadCount > params.executionCount ? params.executionCount - i : params.threadCount;
			qDebug() << "Executions" << i + 1 << "-" << i + executionsCount << "of" << (curCase.zoomFactor == 1 ? "case" : "zoom case") << curCase.problemFileName << "output to" << outputDir.absolutePath();
			QStringList curArguments = QStringList() << arguments << "--parallel=" + QString::number(executionsCount);

			// Start process
			myProcess->start(params.executablePath, curArguments);
			if (!myProcess->waitForFinished(-1)) qDebug() << "Raster Packing failed:" << myProcess->errorString();
			else qDebug().noquote() << "Finished Raster Packing successfully. Output:\n" << myProcess->readAll();
		}
	}
}
