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

	void getIfpBoundingBox(int itemId, int angle, int &bottomLeftX, int &bottomLeftY, int &topRightX, int &topRightY, std::shared_ptr<RasterPackingProblem> problem);

	class RasterStripPackingSolverBase {
	public:
		RasterStripPackingSolverBase() {}

		// Info Functions
		int getNumItems() { return originalProblem->count(); }
		int getCurrentWidth() { return currentWidth; }
		int getCurrentHeight() { return currentHeight; }
		// Basic Functions
		virtual void generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		virtual void generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		// --> Get layout overlap (sum of individual overlap values)
		virtual qreal getGlobalOverlap(RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		// --> Local search
		virtual void performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		// --> Change container size
		virtual bool setContainerWidth(int &pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) = 0;
		virtual void resetWeights() = 0;

	protected:
		virtual void updateMapsLength(int pixelWidth, RasterStripPackingParameters &params) = 0;

		std::shared_ptr<RasterPackingProblem> originalProblem;
		TotalOverlapMapSet maps;
		int currentWidth, initialWidth;
		// FIXME: Should be only in  RasterStripPackingSolver2D solver
		int currentHeight, initialHeight;
	};

	class RasterStripPackingSolver : public RasterStripPackingSolverBase
	{
		friend class MainWindow;

	public:
		RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem) { setProblem(_problem); determineMinimumOriginalIfpWidth(_problem); }

		// Basic Functions
		void generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		virtual void generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Get layout overlap (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Local search
		void performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Change container size
		bool setContainerWidth(int &pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Guided Local Search functions
		void updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) {}
		void resetWeights() {}
		// Cluster only function
		virtual void declusterSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) {};

	protected:
		virtual void updateMapsLength(int pixelWidth, RasterStripPackingParameters &params);

		void setProblem(std::shared_ptr<RasterPackingProblem> _problem);
		void determineMinimumOriginalIfpWidth(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Get two items overlap
		qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		bool detectOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		qreal getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);
		virtual qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);

		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params);
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);

		int getItemMaxX(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);
		bool detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);

		// Debug only functions
		// --> Get layout overlap with individual values (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps, RasterStripPackingParameters &params);
		// --> Get absolute minimum overlap position
		QPoint getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value, PositionChoice placementHeuristic);

		TotalOverlapMapSet maps;
		int minimumOriginalIfpWidth;
	};
}

#endif // RASTERSTRIPPACKINGSOLVER_H
