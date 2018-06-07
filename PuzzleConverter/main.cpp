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

void decomposeConcave(RASTERPACKING::PackingProblem &problem) {
	for (auto it = problem.pbegin(); it != problem.pend(); it++) {
		std::shared_ptr<RASTERPACKING::Piece> curPiece = *it;
		curPiece->decomposeConvex();
		//std::shared_ptr<RASTERPACKING::Polygon> curPol = curPiece->getPolygon();

		//qDebug() << curPiece->getName() << ":";
		//POLYDECOMP::Polygon incPoly;
		//for (auto it2 = curPol->begin(); it2 != curPol->end(); it2++) {
		//	QPointF curPt = *it2;
		//	qDebug() << curPt.x() << curPt.y();
		//	incPoly.push(POLYDECOMP::Point(curPt.x(), curPt.y()));
		//}
		//incPoly.makeCCW();
		//POLYDECOMP::EdgeList diags = incPoly.decomp();
		//std::vector<POLYDECOMP::Polygon> polys = incPoly.slice(incPoly, diags);
		//if (polys.size() > 1) qDebug() << "Concave!" << polys.size() << "concave partitions.";
		//else qDebug() << "Convex!";

		//qDebug() << "";
	}
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
		qWarning() << "Output file already exists!";
	}

    qDebug() << "Program execution started.";
    QTime myTimer;

    qDebug() << "Loading problem file...";
    RASTERPACKING::PackingProblem problem;
    myTimer.start(); totalTimer.start();
	//problem.load(params.inputFilePath);
	problem.load(params.inputFilePath, "terashima");
    qDebug() << "Problem file read." << problem.getNofitPolygonsCount() << "nofit polygons read in" << myTimer.elapsed()/1000.0 << "seconds";
	
	decomposeConcave(problem);

	problem.savePuzzle(params.outputFilePath);

    qDebug() << "Program execution finished. Total time:" << totalTimer.elapsed()/1000.0 << "seconds.";
    return 0;
}
