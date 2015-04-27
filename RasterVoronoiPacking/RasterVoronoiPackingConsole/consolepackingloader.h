#ifndef CONSOLEPACKINGLOADER_H
#define CONSOLEPACKINGLOADER_H

#include <QObject>
#include <QString>
#include "raster/packingthread.h"

namespace RASTERVORONOIPACKING { class RasterStripPackingParameters; }

struct GpuMemoryRequirements {
	GpuMemoryRequirements() {};

	size_t totalIfpMemory;
	size_t totalNfpMemory;
	size_t maxSingleIfpMemory;
};

class ConsolePackingLoader : public QObject
{
	Q_OBJECT

public:
	explicit ConsolePackingLoader(QObject *parent = 0);

	void setParameters(QString inputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams);
	void setParameters(QString inputFilePath, QString zoomedInputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams);
	void run();

public slots :
	void printExecutionStatus(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, qreal zoomscale, int curLength, int minLength);
	void saveMinimumResult(int minLength, qreal scale, qreal zoomScale, int totalItNum, qreal elapsed, uint seed);
	void saveFinalResult(int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime, qreal scale, qreal zoomscale, int length, uint seed);
	void saveFinalLayout(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, qreal scale, qreal length, uint seed);

private:
	bool loadInputFile(QString inputFilePath, std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> problem, bool &loadGPU);
	RASTERVORONOIPACKING::RasterStripPackingParameters algorithmParamsBackup;
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> problem, zoomProblem;
	QString outputTXTFile, outputXMLFile;
	PackingThread singleThreadedPacker;
};

#endif // CONSOLEPACKINGLOADER_H
