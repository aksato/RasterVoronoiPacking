#ifndef RASTEROVERLAPEVALUATORMAPCUDAINC_H
#define RASTEROVERLAPEVALUATORMAPCUDAINC_H

#define ZOOMNEIGHBORHOOD 1
#include "raster/rasteroverlapevaluator.h"
#include "totaloverlapmapcuda.h"
#include "rasterpackingcudaproblem.h"
#include "rasteroverlapevaluatorcudagls.h"

namespace RASTERVORONOIPACKING {

	class RasterTotalOverlapMapEvaluatorCudaIncremental : public RasterTotalOverlapMapEvaluatorCudaGLS
	{
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorCudaIncremental(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorCudaIncremental(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights);

		// --> 
		void signalNewItemPosition(int itemId, int orientation, QPoint newPos);
		void updateWeights(RasterPackingSolution& solution, QVector<quint32>& overlaps, quint32 maxOverlap);

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution& solution);

	private:
		void initializeMaps(std::shared_ptr<RasterPackingProblem> _problem);
		RasterPackingSolution currentSolution;

	};
}

#endif // RASTEROVERLAPEVALUATORMAPCUDAINC_H
