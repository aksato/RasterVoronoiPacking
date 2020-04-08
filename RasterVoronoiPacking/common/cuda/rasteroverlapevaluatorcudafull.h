#ifndef RASTEROVERLAPEVALUATORMAPCUDAFULL_H
#define RASTEROVERLAPEVALUATORMAPCUDAFULL_H

#include "raster/rasteroverlapevaluator.h"
#include "totaloverlapmapcuda.h"
#include "rasterpackingcudaproblem.h"
#include "rasteroverlapevaluatorcudagls.h"
#include "glsweightsetcuda.h"

namespace RASTERVORONOIPACKING {
	struct DeviceRasterNoFitPolygonSet {
		quint32** data;
		int * widths, * heights;
		int * originsX, * originsY, * multipliers;
	};

	struct DeviceRasterPackingSolution {
		int *posX, *posY, * orientations;
	};

	class RasterTotalOverlapMapEvaluatorCudaFull : public RasterTotalOverlapMapEvaluatorCudaGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSetCuda> _glsWeightsCuda);

		// --> Destructor
		~RasterTotalOverlapMapEvaluatorCudaFull();

		// --> Guided Local Search functions
		// --> Update guided local search weights
		void updateWeights(RasterPackingSolution& solution);
		void updateWeights(RasterPackingSolution& solution, QVector<quint32>& overlaps, quint32 maxOverlap);

		// --> Reset guided local search weights
		void resetWeights();

		// --> Signal placement update
		void signalNewItemPosition(int itemId, int orientation, QPoint newPos);

	protected:
		// Access weigths
		int getWeight(int itemId1, int itemId2) { return glsWeightsCuda->getWeight(itemId1, itemId2); }

		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution);
		//std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution, QList<int>& placedItems);

	private:
		// -->  Guided local search weights on GPU
		std::shared_ptr<GlsWeightSetCuda> glsWeightsCuda;

		void initCuda(std::shared_ptr<RasterPackingProblem> _problem);
		DeviceRasterNoFitPolygonSet d_nfps;
		DeviceRasterPackingSolution d_solution;
		int* d_itemId2ItemTypeMap;
		int getRasterNoFitPolygonKey(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId);
		int numAngles, numKeys;
	};
}

#endif // RASTEROVERLAPEVALUATORMAPCUDAFULL_H
