#ifndef RASTERSTRIPPACKINGSOLVERGLS_H
#define RASTERSTRIPPACKINGSOLVERGLS_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasterpackingsolution.h"
#include "rasterstrippackingsolver.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

class MainWindow;

namespace RASTERVORONOIPACKING {

	class RasterStripPackingSolverGLS : public RasterStripPackingSolver
	{
		friend class MainWindow;
	public:
		RasterStripPackingSolverGLS(std::shared_ptr<RasterPackingProblem> _problem) : RasterStripPackingSolver(_problem) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(originalProblem->count())); }

		// --> Guided Local Search functions
		void updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void resetWeights();

	protected:
		qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params);

		// Debug only functions
		// --> Get layout overlap with individual values (sum of individual overlap values)
		std::shared_ptr<GlsWeightSet> getGlsWeights() { return glsWeights; }

		std::shared_ptr<RasterPackingProblem> roughGridProblem;
		std::shared_ptr<GlsWeightSet> glsWeights;
	};


	class RasterStripPackingSolver2D : public RasterStripPackingSolverGLS {
		friend class MainWindow;

	public:
		RasterStripPackingSolver2D(std::shared_ptr<RasterPackingProblem> _problem) : RasterStripPackingSolverGLS(_problem) {
			currentHeight = this->originalProblem->getContainerHeight();
			initialHeight = currentHeight;
		}
		void generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		void generateBottomLeftSquareSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params);
		bool setContainerDimensions(int &pixelWidthX, int &pixelWidthY, RasterPackingSolution &solution, RasterStripPackingParameters &params);
	private:
		int getItemMaxY(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem);
		void updateMapsDimensions(int pixelWidth, int pixelHeight, RasterStripPackingParameters &params);
	};

	class RasterStripPackingSolverClusterGLS : public RasterStripPackingSolverGLS
	{
		friend class MainWindow;
	public:
		RasterStripPackingSolverClusterGLS(std::shared_ptr<RasterPackingClusterProblem> _problem) : RasterStripPackingSolverGLS(_problem) {
			this->clusterProblem = _problem;
		}
		void declusterSolution(RASTERVORONOIPACKING::RasterPackingSolution &solution) { 
			if (solution.getNumItems() == clusterProblem->getOriginalProblem()->count()) return;
			clusterProblem->convertSolution(solution);
		}
	private:
		std::shared_ptr<RasterPackingClusterProblem> clusterProblem;
	};
}

#endif // RASTERSTRIPPACKINGSOLVERGLS_H
