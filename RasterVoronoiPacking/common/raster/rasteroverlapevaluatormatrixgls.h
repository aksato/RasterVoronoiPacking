#ifndef RASTEROVERLAPEVALUATORMATRIXGLS_H
#define RASTEROVERLAPEVALUATORMATRIXGLS_H

#define ZOOMNEIGHBORHOOD 1
#include <iostream>
#include "rasteroverlapevaluator.h"
#include <Eigen/Core>

namespace RASTERVORONOIPACKING {
	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorMatrixGLS : public RasterTotalOverlapMapEvaluatorGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, cuttingStock) {}

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, cuttingStock) {}

		// --> Creates vector of weights
		static void createWeigthVector(int itemId, int numItems, std::shared_ptr<GlsWeightSet> glsWeights, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
	};

	class RasterTotalOverlapMapEvaluatorCudaMatrixGLS : public RasterTotalOverlapMapEvaluatorGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		// --> Set of all item overlap matrices
		TotalOverlapMapSet matrices;
	};
}

#endif // RASTEROVERLAPEVALUATORMATRIXGLS_H
