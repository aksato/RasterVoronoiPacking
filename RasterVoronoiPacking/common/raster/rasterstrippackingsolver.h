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
	enum BottomLeftMode {BL_STRIPPACKING, BL_RECTANGULAR, BL_SQUARE};

	class RasterStripPackingSolver
	{
		friend class MainWindow;

	public:
		RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem) { setProblem(_problem);}

		// Info Functions
		int getNumItems() { return originalProblem->count(); }
		int getCurrentWidth() { return currentWidth; }
		int getCurrentHeight() { return currentHeight; }
		// Basic Functions
		void generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params, BottomLeftMode mode = BL_STRIPPACKING);
		// --> Get layout overlap (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Local search
		void performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Change container size
		bool setContainerWidth(int &pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		bool setContainerDimensions(int &pixelWidthX, int &pixelWidthY, RasterPackingSolution &solution, RasterStripPackingParameters &params);
		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) {};
		virtual void resetWeights() {};
		// --> Cluster
		virtual void declusterSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) {};
		// Size information function
		int getMinimumContainerWidth() { return originalProblem->getMaxWidth(); }
		int getMinimumContainerHeight() { return originalProblem->getMaxHeight(); }

	protected:
		virtual void updateMapsLength(int pixelWidth, RasterStripPackingParameters &params);
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight, RasterStripPackingParameters &params);

		void setProblem(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Get two items overlap
		qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		bool detectOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		qreal getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);
		virtual qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);

		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params);
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);

		int getItemMaxX(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);
		int getItemMaxY(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);

		bool detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);

		// Debug only functions
		// --> Get layout overlap with individual values (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps, RasterStripPackingParameters &params);
		// --> Get absolute minimum overlap position
		QPoint getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value, PositionChoice placementHeuristic);

		std::shared_ptr<RasterPackingProblem> originalProblem;
		TotalOverlapMapSet maps;
		int currentWidth, currentHeight, initialWidth, initialHeight;

	private:
		void generateBottomLeftStripSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void generateBottomLeftRectangleSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void generateBottomLeftSquareSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
	};
}

#endif // RASTERSTRIPPACKINGSOLVER_H
