#ifndef RASTEROVERLAPEVALUATORCUDAMATRIXGLS_H
#define RASTEROVERLAPEVALUATORCUDAMATRIXGLS_H

#include "raster/rasteroverlapevaluator.h"
#include "cuda/rasteroverlapevaluatorcudagls.h"
#include "cuda/totaloverlapmatrixcuda.h"
#include "cuda/glsweightsetcuda.h"

namespace RASTERVORONOIPACKING {
	class RasterTotalOverlapMapEvaluatorCudaMatrixGLS : public RasterTotalOverlapMapEvaluatorCudaGLS
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
		void populateMatrices();
		std::vector<cudaStream_t> streams;
	};
}

#endif // RASTEROVERLAPEVALUATORCUDAMATRIXGLS_H
