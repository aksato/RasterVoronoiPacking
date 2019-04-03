#ifndef RASTEROVERLAPEVALUATORMAPCUDAGLS_H
#define RASTEROVERLAPEVALUATORMAPCUDAGLS_H

#define ZOOMNEIGHBORHOOD 1
#include "raster/rasteroverlapevaluator.h"
#include "totaloverlapmapcuda.h"
#include "rasterpackingcudaproblem.h"

namespace RASTERVORONOIPACKING {

	class RasterTotalOverlapMapEvaluatorCudaGLS : public RasterTotalOverlapMapEvaluator
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaGLS(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);

		// --> TODO: Implementation
		void updateMapsDimensions(int pixelWidth, int pixelHeight) {}

		static std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> getOverlapMapFromDevice(std::shared_ptr<RASTERVORONOIPACKING::TotalOverlapMap> deviceMap);

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

		// --> TODO: Implementation
		std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) { return nullptr; };

		// --> Set of all item overlap matrices
		ItemGeometricToolSet<std::shared_ptr<TotalOverlapMapCuda>> cudamaps;

	private:
		void populateMaps();
	};
}

#endif // RASTEROVERLAPEVALUATORMAPCUDAGLS_H
