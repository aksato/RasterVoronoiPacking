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
#include "dt\dt.h"
#include "dt\pnmfile.h"
#include "dt\imconv.h"
#include <omp.h>

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

bool preProcessProblem(RASTERPACKING::PackingProblem &problem, PreProcessorParameters &params, QString outputPath, QString clusterInfo) {
	QTime myTimer;
	qDebug() << "Innerfit polygons rasterization started.";
	myTimer.start();
	int numProcessed = 1;
	int onePercentIfpCount = qRound(0.01 * (qreal)problem.getInnerfitPolygonsCount());
	std::cout.precision(2);
	for (QList<std::shared_ptr<RASTERPACKING::InnerFitPolygon>>::const_iterator it = problem.cifpbegin(); it != problem.cifpend(); it++, numProcessed++) {
		std::shared_ptr<RASTERPACKING::Polygon> curPolygon = (*it)->getPolygon();
		QPoint referencePoint;

		// --> Rasterize polygon
		int width, height;
		curPolygon->getRasterBoundingBox(referencePoint, params.rasterScaleFactor, width, height);
		std::shared_ptr<RASTERPACKING::RasterInnerFitPolygon> curRasterIFP(new RASTERPACKING::RasterInnerFitPolygon(*it, width, height));
		curRasterIFP->setScale(params.rasterScaleFactor);
		curRasterIFP->setReferencePoint(referencePoint);
		curRasterIFP->setFileName(curPolygon->getName());
		problem.addRasterInnerfitPolygon(curRasterIFP);

		if (onePercentIfpCount == 0 || numProcessed % onePercentIfpCount == 0) {
			qreal progress = (qreal)numProcessed / (qreal)problem.getInnerfitPolygonsCount();
			std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
			int k = 0;
			for (; k < progress * 56; k++) std::cout << "#";
			for (; k < 56; k++) std::cout << ".";
			std::cout << "]";
		}
	}
	std::cout << std::endl;
	qDebug() << "Innerfit polygons rasterization finished." << problem.getInnerfitPolygonsCount() << "polygons processed in" << myTimer.elapsed() / 1000.0 << "seconds.";

	qDebug() << "Nofit polygons rasterization started.";
	myTimer.start();
	
	int onePercentNfpCount = qRound(0.01 * (qreal)problem.getNofitPolygonsCount());
	std::cout.precision(2);
	QVector<QPair<int, int>> imageSizes;
	QVector<QPoint> refPts;
	//QStringList distTransfNames;
	QVector<quint32 *> rasterPolygonVecs;
	QString binFileName = params.outputXMLName;
	binFileName.replace(".xml", ".bin");
	imageSizes.resize(problem.getNofitPolygonsCount());
	refPts.resize(problem.getNofitPolygonsCount());
	rasterPolygonVecs.resize(problem.getNofitPolygonsCount());
	problem.resizeRasterNoFitPolygon();
	numProcessed = 0;
	#pragma omp parallel for
	for (int polygonId = 0; polygonId < problem.getNofitPolygonsCount(); polygonId++) {
		std::shared_ptr<RASTERPACKING::Polygon> curPolygon = problem.getNofitPolygon(polygonId)->getPolygon();
		QPoint referencePoint;

		// --> Rasterize polygon
		int width, height;
		int *rasterCurPolygonVec;
		rasterCurPolygonVec = curPolygon->getRasterImage(referencePoint, params.rasterScaleFactor, width, height);
		
		imageSizes[polygonId] = QPair<int, int>(width, height);
		refPts[polygonId] = referencePoint;
		//distTransfNames.push_back(curPolygon->getName() + ".png");

		if (params.skipDt) {
			quint32 *nonDistTransfotmedVec = new quint32[width*height];
			for (int index = 0; index < width*height; index++)
				nonDistTransfotmedVec[index] = (quint32)rasterCurPolygonVec[index];
			rasterPolygonVecs[polygonId] = nonDistTransfotmedVec;

			if (params.outputImages) {
				QImage rasterCurPolygon = getImageFromVec(rasterCurPolygonVec, width, height);
				rasterCurPolygon.save(outputPath + "/r" + curPolygon->getName() + ".png");
			}
		}
		else {
			// Create dt image
			int index = 0;
			image<uchar> *input = new image<uchar>(width, height);
			for (int i = 0; i < height; i++) {
				unsigned char *data = imPtr(input, 0, i);
				for (int j = 0; j < width; j++)
					data[j] = rasterCurPolygonVec[index++];
			}
			// compute dt
			image<double> *out = dt(input);
			// take square roots
			quint32 *distTransfotmedVec = new quint32[width*height];
			index = 0;
			for (int x = 0; x < out->width(); x++) {
				for (int y = 0; y < out->height(); y++) {
					distTransfotmedVec[index++] = qRound(10 * sqrt(imRef(out, x, y)));
					imRef(out, x, y) = sqrt(imRef(out, x, y));
				}
			}
			rasterPolygonVecs[polygonId] = distTransfotmedVec;

			//if (params.outputImages) {
			//	// convert to grayscale
			//	image<uchar> *gray = imageFLOATtoUCHAR(out);
			//	// save output
			//	QImage result(width, height, QImage::Format_Indexed8);
			//	setColormap(result);
			//	for (int i = 0; i < height; i++)
			//	for (int j = 0; j < width; j++)
			//		result.setPixel(j, i, imRef(gray, j, i));
			//	result.save(outputPath + "/" + curPolygon->getName() + ".png");
			//}

		}
		std::shared_ptr<RASTERPACKING::RasterNoFitPolygon> curRasterNFP(new RASTERPACKING::RasterNoFitPolygon(problem.getNofitPolygon(polygonId), width, height));
		curRasterNFP->setScale(params.rasterScaleFactor);
		curRasterNFP->setReferencePoint(referencePoint);
		curRasterNFP->setFileName(curPolygon->getName());
		problem.addRasterNofitPolygon(curRasterNFP, polygonId);

		#pragma omp atomic
		numProcessed++;

		if (onePercentNfpCount == 0 || numProcessed % onePercentNfpCount == 0 || numProcessed == problem.getNofitPolygonsCount()) {
			#pragma omp critical
			{
				qreal progress = (qreal)numProcessed / (qreal)problem.getNofitPolygonsCount();
				std::cout << "\r" << "Progress : [" << std::fixed << 100.0*progress << "%] [";
				int k = 0;
				for (; k < progress * 56; k++) std::cout << "#";
				for (; k < 56; k++) std::cout << ".";
				std::cout << "]";
			}
		}
	}
	std::cout << std::endl;
	qreal totalArea = 0;
	std::for_each(imageSizes.begin(), imageSizes.end(), [&totalArea](QPair<int, int> &size){totalArea += (qreal)(size.first*size.second); });
	qDebug() << "Nofit polygons rasterization finished." << problem.getNofitPolygonsCount() << "polygons processed in" << myTimer.elapsed() / 1000.0 << "seconds. Total area:" << totalArea;

	qDebug() << "Saving output files.";
	myTimer.start();
	// Save binary file
	QFile file(outputPath + QDir::separator() + binFileName);
	file.open(QIODevice::WriteOnly);
	QDataStream out(&file);   // we will serialize the data into the file
	out.setByteOrder(QDataStream::LittleEndian);
	// Print polygon count
	out << (qint32)problem.getNofitPolygonsCount();
	// Output polygon sizes and reference points
	for (int i = 0; i < problem.getNofitPolygonsCount(); i++) {
		out << (qint32)imageSizes[i].first << (qint32)imageSizes[i].second << (qint32)refPts[i].x() << (qint32)refPts[i].y();
	}
	// Print polygon data sequentially
	for (int i = 0; i < problem.getNofitPolygonsCount(); i++) {
		quint32 *curPol = rasterPolygonVecs[i];
		for (int k = 0; k < imageSizes[i].first*imageSizes[i].second; k++) out << curPol[k];
	}
	file.close();
	problem.save(outputPath + QDir::separator() + params.outputXMLName, binFileName, clusterInfo);
	qDebug() << "Output files saved in" << myTimer.elapsed() / 1000.0 << "seconds";

	return true;
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
	if (!QDir(params.outputDir).exists()) {
		qWarning() << "Destination folder does not exist. Creating it.";
		QDir(params.outputDir).mkpath(params.outputDir);
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
	preProcessProblem(problem, params, params.outputDir, "");

	RASTERPACKING::PackingProblem clusterProblem;
	QString clusterInfo;
	
    qDebug() << "Program execution finished. Total time:" << totalTimer.elapsed()/1000.0 << "seconds.";
    return 0;
}
