#ifndef RASTEROVERLAPEVALUATORMATRIXGLS_H
#define RASTEROVERLAPEVALUATORMATRIXGLS_H

#define ZOOMNEIGHBORHOOD 1

#include "rasteroverlapevaluator.h"
#include <Eigen/Core>

namespace RASTERVORONOIPACKING {
	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorMatrixGLS : public RasterTotalOverlapMapEvaluatorGLS
	{
		friend class ::MainWindow;
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, cuttingStock) { ; }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, cuttingStock) {}

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

	private:
		void createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);
	};
}

#endif // RASTEROVERLAPEVALUATORMATRIXGLS_H
