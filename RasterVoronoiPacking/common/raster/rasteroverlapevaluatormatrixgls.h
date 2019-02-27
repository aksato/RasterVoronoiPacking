#ifndef RASTEROVERLAPEVALUATORMATRIXGLS_H
#define RASTEROVERLAPEVALUATORMATRIXGLS_H

#define ZOOMNEIGHBORHOOD 1

#include "rasteroverlapevaluator.h"
#include <Eigen/Core>

namespace RASTERVORONOIPACKING {
	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorMatrixGLS : public RasterTotalOverlapMapEvaluatorGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, cuttingStock) { ; }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, cuttingStock) {}

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		// --> Creates vector of weights
		void createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);
	};

	class RasterTotalOverlapMapEvaluatorCudaMatrixGLS : public RasterTotalOverlapMapEvaluatorMatrixGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorMatrixGLS(_problem, cuttingStock) { ; }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorMatrixGLS(_problem, _glsWeights, cuttingStock) {}

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
	};
}

#endif // RASTEROVERLAPEVALUATORMATRIXGLS_H
