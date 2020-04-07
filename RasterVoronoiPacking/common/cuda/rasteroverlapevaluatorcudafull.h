#ifndef RASTEROVERLAPEVALUATORMAPCUDAFULL_H
#define RASTEROVERLAPEVALUATORMAPCUDAFULL_H

#include "raster/rasteroverlapevaluator.h"
#include "totaloverlapmapcuda.h"
#include "rasterpackingcudaproblem.h"
#include "rasteroverlapevaluatorcudagls.h"

namespace RASTERVORONOIPACKING {
	struct DeviceRasterNoFitPolygonSet {
		quint32** d_data;
		int * d_widths, * d_heights;
		int * d_originsX, * d_originsY, * d_multipliers;
	};

	struct DeviceRasterPackingSolution {
		int *d_posX, *d_posY, * d_orientations;
	};

	class RasterTotalOverlapMapEvaluatorCudaFull : public RasterTotalOverlapMapEvaluatorCudaGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaFull(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights);

		// --> Destructor
		~RasterTotalOverlapMapEvaluatorCudaFull();

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution);
		//std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution, QList<int>& placedItems);

	private:
		void initCuda(std::shared_ptr<RasterPackingProblem> _problem);
		DeviceRasterNoFitPolygonSet d_nfps;
		int* d_itemId2ItemTypeMap;
		int getRasterNoFitPolygonKey(int staticPieceTypeId, int staticAngleId, int orbitingPieceTypeId, int orbitingAngleId);
		int numAngles, numKeys;
	};
}

#endif // RASTEROVERLAPEVALUATORMAPCUDAFULL_H
