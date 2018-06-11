#include <QtCore/QCoreApplication>
#include <QDebug>
#include <QImage>
#include <QCommandLineParser>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QTime>
#include <iostream>
#include "colormap.h"
#include "parametersParser.h"
#include "../RasterVoronoiPacking/common/packingproblem.h"
#include "polydecomp-keil/polygon.h"

QTime totalTimer;

int decomposeConcave(RASTERPACKING::PackingProblem &problem) {
	int concaveCount = 0;
	for (auto it = problem.pbegin(); it != problem.pend(); it++) {
		std::shared_ptr<RASTERPACKING::Piece> curPiece = *it;
		concaveCount += curPiece->decomposeConvex();
	}
	return concaveCount;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("Puzzle Converter");
    QCoreApplication::setApplicationVersion("0.1");

    // --> Parse command line arguments
    QCommandLineParser parser;
    parser.setApplicationDescription("Puzzle Converter from ESICUP to puzzle.");
    PreProcessorParameters params;
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

    // Transform source and destination paths to absolute
    QFileInfo inputFileInfo(params.inputFilePath);
    params.inputFilePath = inputFileInfo.absoluteFilePath();
	QFileInfo outputFileInfo(params.outputFilePath);
	params.outputFilePath = outputFileInfo.absoluteFilePath();

    // Check if source and destination paths exist
    if(!QFile(params.inputFilePath).exists()) {
        qCritical() << "Input file not found.";
        return 1;
    }
	if (QFile(params.outputFilePath).exists()) {
		qWarning() << "Warning: output file already exists!";
	}

    qDebug() << "Program execution started.";
    QTime myTimer;

    qDebug() << "Loading problem file...";
    RASTERPACKING::PackingProblem problem;
    myTimer.start(); totalTimer.start();

	if(!params.terashima)
		problem.load(params.inputFilePath);
	else {
		// Read text file
		QFile file(params.inputFilePath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) { qCritical() << "Input file not found"; return 0; }
		QTextStream f(&file);
		problem.loadTerashima(f,params.subProblem);
	}
    qDebug() << "Problem file read.";
	
	int concaveCount = decomposeConcave(problem);

	problem.savePuzzle(params.outputFilePath);

    qDebug() << "Program execution finished. Decomposed" << concaveCount << "concave polygons. Total time:" << totalTimer.elapsed()/1000.0 << "seconds.";
    return 0;
}
