#ifndef RASTERSTRIPPACKINGSOLVER_H
#define RASTERSTRIPPACKINGSOLVER_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasterpackingsolution.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

class MainWindow;

namespace RASTERVORONOIPACKING {

	class RasterStripPackingSolver
	{
		friend class MainWindow;

	public:
		RasterStripPackingSolver() {}
		RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem) { setProblem(_problem); }
		void setProblem(std::shared_ptr<RasterPackingProblem> _problem, bool isZoomedProblem = false);

		// Info Functions
		int getNumItems() { return originalProblem->count(); }
		int getCurrentWidth() { return currentWidth; }

		// Basic Functions
		void generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Get layout overlap (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Local search
		void performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Guided Local Search functions
		void updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void resetWeights();
		// --> Retrieve a rectangular area of the total overlap map
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, bool useGlsWeights = false);

		// Auxiliary functions
		// --> Change container size
		void setContainerWidth(int pixelWidth);
		void setContainerWidth(int pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params);

	private:
		// --> Get two items overlap
		qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		qreal getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);
		qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem, bool useGlsWeights = false);
		void updateItemCacheInfo(int itemId, QPoint oldPos, int oldAngle, bool useGlsWeights = false);

		void performLocalSearchSingleResolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void performLocalSearchDoubleResolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params);
		QPoint getMinimumOverlapPositionGPU(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params);
		QPoint getZoomedMinimumOverlapPosition(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMapSerial(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params); // Private
		void getTotalOverlapMapSerialNoCacheNoWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution);
		void getTotalOverlapMapSerialNoCacheWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution);
		void getTotalOverlapMapSerialCacheNoWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution);
		void getTotalOverlapMapSerialCacheWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution);
		void updateItemCacheInfo(int itemId, QPoint oldPos, int oldAngle, RasterStripPackingParameters &params);

		// Debug only functions
		// --> Get layout overlap with individual values (sum of individual overlap values)
		std::shared_ptr<GlsWeightSet> getGlsWeights() { return glsWeights; }
		qreal getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps, RasterStripPackingParameters &params);
		void printCompleteCacheInfo(int itemId, int orientation, bool useGlsWeights = false);
		// --> Return total overlap map for a given item
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMapGPU(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights = false);
		// --> Get absolute minimum overlap position
		QPoint getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value);

		//std::shared_ptr<RasterPackingProblem> currentProblem;
		std::shared_ptr<RasterPackingProblem> originalProblem;
		std::shared_ptr<RasterPackingProblem> zoomedProblem;
		TotalOverlapMapSet maps;
		std::shared_ptr<GlsWeightSet> glsWeights;
		int currentWidth, initialWidth;
	};

}

#endif // RASTERSTRIPPACKINGSOLVER_H
