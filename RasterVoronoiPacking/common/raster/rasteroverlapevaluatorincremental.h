#ifndef RASTEROVERLAPEVALUATORINCREMENTAL_H
#define RASTEROVERLAPEVALUATORINCREMENTAL_H

#include "rasteroverlapevaluator.h"

namespace RASTERVORONOIPACKING {

	// --> Full overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorIncremental : public RasterTotalOverlapMapEvaluatorGLS
	{
		friend class ::MainWindow;

	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorIncremental(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorIncremental(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false);

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

#endif