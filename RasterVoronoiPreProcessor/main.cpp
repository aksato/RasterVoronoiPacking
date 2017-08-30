
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
#include "cuda_runtime.h"

cudaError_t transform( int * S, float * D, int N, int M);

QImage getImageFromVec(int *S, int width, int height) {
    QImage result(width, height, QImage::Format_Mono);
    result.setColor(1, qRgb(255, 255, 255));
    result.setColor(0, qRgb(255, 0, 0));
    result.fill(0);
    for (int i=0; i<height; i++)
        for (int j=0; j<width; j++)
            result.setPixel(j, i, S[i*width+j]);
    return result;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("Raster Packing Preprocessor");
    QCoreApplication::setApplicationVersion("0.1");

    // --> Parse command line arguments
    QCommandLineParser parser;
    parser.setApplicationDescription("Raster Voronoi nofit polygon generator.");
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
    QDir outputDirDir(params.outputDir);
    params.outputDir = outputDirDir.absolutePath();

    // Check if source and destination paths exist
    if(!QFile(params.inputFilePath).exists()) {
        qCritical() << "Input file not found.";
        return 1;
    }
    if(!QDir(params.outputDir).exists()) {
        QTextStream stream(stdin);
        bool ok = false;
        while(!ok) {
            std::cout << "Destination folder does not exist. Would you like to create it [Y/n]? ";
            QString line = stream.readLine().toLower().trimmed();
            if(line == "" || line == "y" || line == "yes") {
                QDir(params.outputDir).mkpath(params.outputDir);
                ok = true;
            }
            else if (line == "n" || line == "no") {
                qCritical() << "Program execution stopped.";
                return 1;
            }
        }
    }

    if(params.optionsFile != "") {
        if(!QFile(params.optionsFile).exists()) {
            qWarning() << "Warning: options file not found. Parameters set to default.";
        }
        else {
            switch (parseOptionsFile(params.optionsFile, &params, &errorMessage)) {
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
            QFileInfo optionsFileInfo(params.optionsFile);
            QDir::setCurrent(optionsFileInfo.absolutePath());
        }
    }

    qDebug() << "Program execution started.";
    QTime myTimer, totalTimer;

    qDebug() << "Loading problem file...";
    RASTERPACKING::PackingProblem problem;
    myTimer.start(); totalTimer.start();
	problem.load(params.inputFilePath, params.inputFileType, params.puzzleScaleFactor, params.scaleFixFactor);
    if(params.headerFile != "") problem.copyHeader(params.headerFile);
    qDebug() << "Problem file read." << problem.getNofitPolygonsCount() << "nofit polygons read in" << myTimer.elapsed()/1000.0 << "seconds";

	qDebug() << "Innerfit polygons rasterization started.";
	myTimer.start();
	int numProcessed = 1;
	std::cout.precision(2);
	for(QList<std::shared_ptr<RASTERPACKING::InnerFitPolygon>>::const_iterator it = problem.cifpbegin(); it != problem.cifpend(); it++, numProcessed++) {
        std::shared_ptr<RASTERPACKING::Polygon> curPolygon = (*it)->getPolygon();

		QPoint referencePoint;

		// --> Rasterize polygon
		int width, height, *rasterCurPolygonVec;
		if (params.innerFitEpsilon < 0) rasterCurPolygonVec = curPolygon->getRasterImageVector(referencePoint, params.rasterScaleFactor, width, height);
		else rasterCurPolygonVec = curPolygon->getRasterBoundingBoxImageVector(referencePoint, params.rasterScaleFactor, params.innerFitEpsilon, width, height);
        QImage rasterCurPolygon = getImageFromVec(rasterCurPolygonVec, width, height);
        rasterCurPolygon.save(params.outputDir + "/" + curPolygon->getName() + ".png");

		std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIFP(new RASTERPACKING::RasterInnerFitPolygon(*it));
        curRasterIFP->setScale(params.rasterScaleFactor);
        curRasterIFP->setReferencePoint(referencePoint);
        curRasterIFP->setFileName(curPolygon->getName() + ".png");
		problem.addRasterInnerfitPolygon(curRasterIFP);

		qreal progress = (qreal)numProcessed/(qreal)problem.getInnerfitPolygonsCount();
        std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
        int k = 0;
        for(; k < progress*56; k++) std::cout << "#";
        for(; k < 56; k++) std::cout << ".";
        std::cout << "]";

		delete[] rasterCurPolygonVec;
	}
	std::cout << std::endl;
	qDebug() << "Innerfit polygons rasterization finished." << problem.getInnerfitPolygonsCount() << "polygons processed in" << myTimer.elapsed()/1000.0 << "seconds.";

    qDebug() << "Nofit polygons rasterization started.";
	myTimer.start();
	numProcessed = 1;
	std::cout.precision(2);
	QVector<QPair<int,int>> imageSizes;
    QStringList distTransfNames;
    QVector<int *> rasterPolygonVecs;
    for(QList<std::shared_ptr<RASTERPACKING::NoFitPolygon>>::const_iterator it = problem.cnfpbegin(); it != problem.cnfpend(); it++, numProcessed++) {
        std::shared_ptr<RASTERPACKING::Polygon> curPolygon = (*it)->getPolygon();

        QPoint referencePoint;

        // --> Rasterize polygon
        int width, height;
        //int *rasterCurPolygonVec = curPolygon->getRasterImageVector(referencePoint, params.rasterScaleFactor, width, height);
		int *rasterCurPolygonVec;
		if(!params.noOverlap)
			rasterCurPolygonVec = curPolygon->getRasterImageVector(referencePoint, params.rasterScaleFactor, width, height);
		else
			rasterCurPolygonVec = curPolygon->getRasterImageVectorWithContour(referencePoint, params.rasterScaleFactor, width, height);
        if(params.saveRaster) {
            QImage rasterCurPolygon = getImageFromVec(rasterCurPolygonVec, width, height);
            rasterCurPolygon.save(params.outputDir + "/r" + curPolygon->getName() + ".png");
        }
		imageSizes.push_back(QPair<int,int>(width,height));
        distTransfNames.push_back(curPolygon->getName() + ".png");
		rasterPolygonVecs.push_back(rasterCurPolygonVec);

		std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> curRasterNFP(new RASTERPACKING::RasterNoFitPolygon(*it));
        curRasterNFP->setScale(params.rasterScaleFactor);
        curRasterNFP->setReferencePoint(referencePoint);
        curRasterNFP->setFileName(curPolygon->getName() + ".png");
        problem.addRasterNofitPolygon(curRasterNFP);

        qreal progress = (qreal)numProcessed/(qreal)problem.getNofitPolygonsCount();
        std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
        int k = 0;
        for(; k < progress*56; k++) std::cout << "#";
        for(; k < 56; k++) std::cout << ".";
        std::cout << "]";
	}
	std::cout << std::endl;
	float totalArea = 0;
	std::for_each(imageSizes.begin(), imageSizes.end(), [&totalArea](QPair<int,int> &size){totalArea += (float)(size.first*size.second);});
	qDebug() << "Nofit polygons rasterization finished." << problem.getNofitPolygonsCount() << "polygons processed in" << myTimer.elapsed()/1000.0 << "seconds. Total area:" << totalArea;

	qDebug() << "Max distance determination started.";
	myTimer.start();
    numProcessed = 1;
    float maxD = 0;
	for(QVector<int *>::const_iterator it = rasterPolygonVecs.begin(); it != rasterPolygonVecs.end(); it++, numProcessed++) {
		int width, height;
		width = imageSizes[numProcessed-1].first;
		height = imageSizes[numProcessed-1].second;
		int *rasterCurPolygonVec  = *it;

		// --> Determine the distance transform and save temporary result
        float *distTransfCurPolygonVec = new float[height*width];
        cudaError_t cudaStatus = transform(rasterCurPolygonVec, distTransfCurPolygonVec, height, width);
        for (int c=0; c < height*width; c++)
            if(distTransfCurPolygonVec[c] > maxD)
                maxD = distTransfCurPolygonVec[c];

        qreal progress = (qreal)numProcessed/(qreal)problem.getNofitPolygonsCount();
        std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
        int k = 0;
        for(; k < progress*56; k++) std::cout << "#";
        for(; k < 56; k++) std::cout << ".";
        std::cout << "]";

        delete[] distTransfCurPolygonVec;
	}
	std::cout << std::endl;
	qDebug() << "Max distance determination finished." << problem.getNofitPolygonsCount() << "polygons processed in" << myTimer.elapsed()/1000.0 << "seconds";

	for(QList<std::shared_ptr<RASTERPACKING::RasterNoFitPolygon>>::iterator it = problem.rnfpbegin(); it != problem.rnfpend(); it++) (*it)->setMaxD(maxD);
	problem.save(params.outputDir + "/" + params.outputXMLName);

	qDebug() << "Final distance transformation started.";
	myTimer.start();
    numProcessed = 1;
	bool dtok = true;
	for(QVector<int *>::const_iterator it = rasterPolygonVecs.begin(); it != rasterPolygonVecs.end(); it++, numProcessed++) {
		int width, height;
		width = imageSizes[numProcessed-1].first;
		height = imageSizes[numProcessed-1].second;
		int *rasterCurPolygonVec  = *it;

		float *distTransfCurPolygonVec = new float[height*width];
        cudaError_t cudaStatus = transform(rasterCurPolygonVec, distTransfCurPolygonVec, height, width);

		QImage result(width, height, QImage::Format_Indexed8);
        setColormap(result);

        for (int i=0; i<height; i++)
            for (int j=0; j<width; j++){
                if(qFuzzyCompare((float)(1 + 0.0), (float)(1 + distTransfCurPolygonVec[i*width+j]))) {
                    result.setPixel(j, i, 0);
                }
                else {
                    //int index = (int)( (distTransfCurPolygonVec[i*width+j]-sqrt(2))*254/(maxD-sqrt(2)) + 1);
					int index = (int)( (distTransfCurPolygonVec[i*width+j]-1)*254/(maxD-1) + 1);
                    if(index < 0 || index > 255) {
                        result.setPixel(j, i, 255);
                        dtok = false;
                    }
                    else result.setPixel(j, i, index);
                }
            }

        result.save(params.outputDir + "/" + distTransfNames.at(numProcessed-1));

		qreal progress = (qreal)numProcessed/(qreal)problem.getNofitPolygonsCount();
        std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
        int k = 0;
        for(; k < progress*56; k++) std::cout << "#";
        for(; k < 56; k++) std::cout << ".";
        std::cout << "]";

		delete[] rasterCurPolygonVec;
        delete[] distTransfCurPolygonVec;

	}
	std::cout << std::endl;
	if(!dtok) qWarning() << "Warning: Some polygons were not correctly processed.";
	qDebug() << "Final distance transformation finished." << problem.getNofitPolygonsCount() << "polygons processed in" << myTimer.elapsed()/1000.0 << "seconds";

    qDebug() << "Program execution finished. Total time:" << totalTimer.elapsed()/1000.0 << "seconds.";
	//return app.exec();
    return 0;
}
