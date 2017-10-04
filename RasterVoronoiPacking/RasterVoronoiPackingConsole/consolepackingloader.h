#ifndef CONSOLEPACKINGLOADER_H
#define CONSOLEPACKINGLOADER_H

#include <QObject>
#include <QString>
#include "raster/packingthread.h"
#include "raster/packing2dthread.h"

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

	void setParameters(QString inputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams, bool appendSeed = false);
	void setParameters(QString inputFilePath, QString zoomedInputFilePath, QString outputTXTFile, QString outputXMLFile, RASTERVORONOIPACKING::RasterStripPackingParameters &algorithmParams, bool appendSeed = false);
	//void run(PackingThread &threadedPacker);
	void run();

public slots :
	void printExecutionStatus(int curLength, int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed);
	void saveMinimumResult(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal elapsed, uint threadSeed);
	void saveFinalResult(const RASTERVORONOIPACKING::RasterPackingSolution &bestSolution, int length, int totalIt, qreal  curOverlap, qreal minOverlap, qreal totalTime, uint seed);
	void updateUnclusteredProblem(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed);
	void threadFinished();
	void saveMinimumResult2D(const RASTERVORONOIPACKING::RasterPackingSolution &solution, const Solution2DInfo &info, int totalItNum, qreal elapsed, uint threadSeed);

signals:
	void quitApp();

private:
	bool loadInputFile(QString inputFilePath, std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> *problem);
	void writeNewLength(int length, int totalItNum, qreal elapsed, uint threadSeed);
	void writeNewLength2D(const Solution2DInfo &info, int totalItNum, qreal elapsed, uint threadSeed);
	void saveXMLSolution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, uint seed);
	RASTERVORONOIPACKING::RasterStripPackingParameters algorithmParamsBackup;
	std::shared_ptr<RASTERVORONOIPACKING::RasterPackingProblem> problem, zoomProblem;
	QString outputTXTFile, outputXMLFile;
	QMap<int, QString*> outlogContents;
	bool appendSeedToOutputFiles;
	int numProcesses;
	QVector<std::shared_ptr<PackingThread>> threadVector;
	QVector<QPair<std::shared_ptr<RASTERVORONOIPACKING::RasterPackingSolution>, qreal>> solutionsCompilation;
	QMap<uint, QList<Solution2DInfo>> solutionInfoHistory;
	qreal totalArea;
	int containerWidth;
};

#endif // CONSOLEPACKINGLOADER_H
