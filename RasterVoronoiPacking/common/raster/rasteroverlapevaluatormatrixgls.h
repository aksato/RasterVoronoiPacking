#ifndef RASTEROVERLAPEVALUATORMATRIXGLS_H
#define RASTEROVERLAPEVALUATORMATRIXGLS_H

#define ZOOMNEIGHBORHOOD 1
#include "rasteroverlapevaluator.h"
#include "totaloverlapmatrix.h"
#include <Eigen/Core>
#include  "cuda/glsweightsetcuda.h"

namespace RASTERVORONOIPACKING {
	class RasterTotalOverlapMapEvaluatorMatrixGLSBase : public RasterTotalOverlapMapEvaluator
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorMatrixGLSBase(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluator(_problem, cuttingStock) {}

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorMatrixGLSBase(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluator(_problem, _glsWeights, cuttingStock) {}

		// --> Container size update functions
		virtual void updateMapsLength(int pixelWidth) = 0;
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight) = 0;

	protected:
		// --> Creates vector of weights
		void createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);

		// --> Create overlap map from matrix
		void createWeigthVector(int itemId, int numItems, std::shared_ptr<GlsWeightSet> glsWeights, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);

	};

	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorMatrixGLS : public RasterTotalOverlapMapEvaluatorMatrixGLSBase
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
	};

	class RasterTotalOverlapMapEvaluatorCudaMatrixGLS : public RasterTotalOverlapMapEvaluator
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaMatrixGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSetCuda> _glsWeightsCuda, bool cuttingStock = false);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);

		// --> TODO: Implementation
		void updateMapsDimensions(int pixelWidth, int pixelHeight) {}

		// --> Guided Local Search functions
		// --> Update guided local search weights
		void updateWeights(RasterPackingSolution &solution);
		void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap);

		// --> Reset guided local search weights
		void resetWeights();

	protected:
		// Access weigths
		int getWeight(int itemId1, int itemId2) { return glsWeightsCuda->getWeight(itemId1, itemId2); }

		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

		// --> TODO: Implementation
		std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) { return nullptr; };

		// --> Set of all item overlap matrices
		ItemGeometricToolSet<std::shared_ptr<TotalOverlapMatrixCuda>> matrices;
		
		// -->  Guided local search weights on GPU
		std::shared_ptr<GlsWeightSetCuda> glsWeightsCuda;

	private:
		void populateMaps();

		// --> Creates vector of weights. tODO: Remove
		void createWeigthVector(int itemId, Eigen::Matrix< unsigned int, Eigen::Dynamic, 1 > &vec);
	};
}

#endif // RASTEROVERLAPEVALUATORMATRIXGLS_H
