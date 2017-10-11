#ifndef RASTERSTRIPPACKINGSOLVER_H
#define RASTERSTRIPPACKINGSOLVER_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasteroverlapevaluator.h"
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
		static std::shared_ptr<RasterStripPackingSolver> createRasterPackingSolver(std::vector<std::shared_ptr<RasterPackingProblem>> problems, RasterStripPackingParameters &parameters, int initialWidth = -1, int initialHeight = -1);

		RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterOverlapEvaluator> _overlapEvaluator);

		// Info Functions
		int getNumItems() { return originalProblem->count(); }
		int getCurrentWidth() { return currentWidth; }
		int getCurrentHeight() { return currentHeight; }
		// Basic Functions
		void generateRandomSolution(RasterPackingSolution &solution);
		void generateBottomLeftSolution(RasterPackingSolution &solution, BottomLeftMode mode = BL_STRIPPACKING);
		// --> Get layout overlap (sum of individual overlap values)
		qreal getGlobalOverlap(RasterPackingSolution &solution);
		// --> Local search
		void performLocalSearch(RasterPackingSolution &solution);
		// --> Change container size
		bool setContainerWidth(int &pixelWidth, RasterPackingSolution &solution);
		bool setContainerWidth(int &pixelWidth);
		bool setContainerDimensions(int &pixelWidthX, int &pixelWidthY, RasterPackingSolution &solution);
		bool setContainerDimensions(int &pixelWidthX, int &pixelWidthY);
		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) { overlapEvaluator->updateWeights(solution); };
		virtual void resetWeights() { overlapEvaluator->resetWeights(); };
		// Size information function
		int getMinimumContainerWidth() { return originalProblem->getMaxWidth(); }
		int getMinimumContainerHeight() { return originalProblem->getMaxHeight(); }

	protected:
		void setProblem(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Get two items overlap
		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value);

		int getItemMaxX(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);
		int getItemMaxY(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);

		bool detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);

		// Debug only functions
		// --> Get absolute minimum overlap position
		QPoint getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value, PositionChoice placementHeuristic);

		std::shared_ptr<RasterPackingProblem> originalProblem;
		int currentWidth, currentHeight, initialWidth, initialHeight;
		std::shared_ptr<RasterOverlapEvaluator> overlapEvaluator;

	private:
		void generateBottomLeftStripSolution(RasterPackingSolution &solution);
		void generateBottomLeftRectangleSolution(RasterPackingSolution &solution);
		void generateBottomLeftSquareSolution(RasterPackingSolution &solution);
	};

	class RasterStripPackingClusterSolver : public RasterStripPackingSolver {
	public:
		RasterStripPackingClusterSolver(std::shared_ptr<RasterPackingClusterProblem> _problem, std::shared_ptr<RasterOverlapEvaluator> _overlapEvaluator) : RasterStripPackingSolver(_problem, _overlapEvaluator) { this->originalClusterProblem = _problem; }
		// --> Cluster
		void declusterSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) {
			if (solution.getNumItems() == originalClusterProblem->getOriginalProblem()->count()) return;
			originalClusterProblem->convertSolution(solution);
		};
	private:
		std::shared_ptr<RasterPackingClusterProblem> originalClusterProblem;
	};
}

#endif // RASTERSTRIPPACKINGSOLVER_H
