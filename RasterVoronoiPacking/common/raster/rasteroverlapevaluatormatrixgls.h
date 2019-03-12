#ifndef RASTEROVERLAPEVALUATORMATRIXGLS_H
#define RASTEROVERLAPEVALUATORMATRIXGLS_H

#define ZOOMNEIGHBORHOOD 1
#include "rasteroverlapevaluator.h"
#include "totaloverlapmatrix.h"
#include <Eigen/Core>
#include  "cuda/glsweightsetcuda.h"
#include  "cuda/totaloverlapmatrixcuda.h"

namespace RASTERVORONOIPACKING {
	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorMatrixGLS : public RasterTotalOverlapMapEvaluator
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);

		// --> TODO: Implementation
		void updateMapsDimensions(int pixelWidth, int pixelHeight) {}

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

		// --> TODO: Implementation
		std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) { return nullptr; };

		// --> Set of all item overlap matrices
		ItemGeometricToolSet<std::shared_ptr<TotalOverlapMatrix>> matrices;

	private:
		// --> Creates vector of weights
		void createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);

		// --> Create overlap map from matrix
		void createWeigthVector(int itemId, int numItems, std::shared_ptr<GlsWeightSet> glsWeights, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);
	};
}

#endif // RASTEROVERLAPEVALUATORMATRIXGLS_H
