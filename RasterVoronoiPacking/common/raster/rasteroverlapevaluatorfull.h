#ifndef RASTEROVERLAPEVALUATORFULL_H
#define RASTEROVERLAPEVALUATORFULL_H

#include "rasteroverlapevaluator.h"
#include "rasterpackingproblem.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

namespace RASTERVORONOIPACKING {

	// --> Full overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorFull : public RasterTotalOverlapMapEvaluatorGLS
	{
		friend class ::MainWindow;

	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorFull(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorFull(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false);

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);
	};
}

#endif // RASTEROVERLAPEVALUATORFULL_H
