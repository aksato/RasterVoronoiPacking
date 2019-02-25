#ifndef RASTERSTRIPPACKINGSOLVER_H
#define RASTERSTRIPPACKINGSOLVER_H
#include <memory>
#include "rasterpackingproblem.h"
#include "rasteroverlapevaluator.h"
#include "rasterpackingsolution.h"
#include "rasterstrippackingparameters.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

//class MainWindow;

namespace RASTERVORONOIPACKING {

	class RasterStripPackingSolver
	{
                friend class ::MainWindow;

	public:
		static std::shared_ptr<RasterStripPackingSolver> createRasterPackingSolver(std::shared_ptr<RasterPackingProblem> problem, RasterStripPackingParameters &parameters);

		RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator);

		// --> Getter Functions
		int getNumItems() { return originalProblem->count(); }
		// Basic Functions
		//void generateRandomSolution(RasterPackingSolution &solution);
		// --> Get layout overlap (sum of individual overlap values)
		quint32 getGlobalOverlap(RasterPackingSolution &solution);
		quint32 getGlobalOverlap(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 &maxOverlap);
		// --> Local search
		void performLocalSearch(RasterPackingSolution &solution);
		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) { overlapEvaluator->updateWeights(solution); };
		virtual void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) { overlapEvaluator->updateWeights(solution, overlaps, maxOverlap); };
		virtual void resetWeights() { overlapEvaluator->resetWeights(); };
		// --> Size information function
		int getMinimumContainerWidth() { return originalProblem->getMaxWidth(); }
		int getMinimumContainerHeight() { return originalProblem->getMaxHeight(); }
		std::shared_ptr<RasterTotalOverlapMapEvaluator> getOverlapEvaluator() { return overlapEvaluator; }

	protected:
		// --> Overlap determination functions
		// Detect if item is in overlapping position for a subset of fixed items
		bool detectItemTotalOverlap(int itemId, RasterPackingSolution &solution);
		quint32 getItemTotalOverlap(int itemId, RasterPackingSolution &solution);

		// --> Pointer to problem, size variables and total map evaluator
		std::shared_ptr<RasterPackingProblem> originalProblem;
		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator;
	};
}

#endif // RASTERSTRIPPACKINGSOLVER_H
