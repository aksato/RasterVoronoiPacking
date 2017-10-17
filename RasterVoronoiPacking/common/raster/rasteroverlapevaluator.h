#ifndef RASTEROVERLAPEVALUATOR_H
#define RASTEROVERLAPEVALUATOR_H

#define ZOOMNEIGHBORHOOD 3

#include "rasterpackingproblem.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

class MainWindow;

namespace RASTERVORONOIPACKING {

	// --> Class that performs determine overlap operations for solver
	class RasterTotalOverlapMapEvaluator
	{
	public:
		// --> Default constructor
		RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem);

		// --> Determines the item total overlap map
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) = 0;
		virtual void resetWeights() = 0;
		virtual std::shared_ptr<GlsWeightSet> getGlsWeights() { return nullptr; }

		// --> Container size update functions
		virtual void updateMapsLength(int pixelWidth);
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight);

	protected:
		// --> Pointer to problem
		std::shared_ptr<RasterPackingProblem> problem;

		// --> Set of all item total overlap maps
		TotalOverlapMapSet maps;
	};

	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorGLS : public RasterTotalOverlapMapEvaluator
	{
		friend class MainWindow;

	public:
		// --> Default constructor
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem) : RasterTotalOverlapMapEvaluator(_problem) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); }

		// --> Constructor using a custom weights
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluator(_problem), glsWeights(_glsWeights) {}

		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		
		// --> Update guided local search weights
		void updateWeights(RasterPackingSolution &solution);

		// --> Reset guided local search weights
		void resetWeights();

	protected:
		// Guided local search weights
		std::shared_ptr<GlsWeightSet> glsWeights;

	private:
		// Debug functions
		void setgetGlsWeights(std::shared_ptr<GlsWeightSet> _glsWeights) { this->glsWeights = _glsWeights; }
		std::shared_ptr<GlsWeightSet> getGlsWeights() { return glsWeights; }
		
	};

	// --> Overlap evaluator class with dual resolution approach and support for guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorDoubleGLS : public RasterTotalOverlapMapEvaluatorGLS
	{
		friend class MainWindow;

	public:
		//RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem), searchMethod(DOUBLE_ROUND) { this->problem = _problem; this->searchProblem = _searchProblem; }
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem, DoubleResolutionMethod _searchMethod) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem), searchMethod(_searchMethod) { this->problem = _problem; this->searchProblem = _searchProblem; }
		//RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem, _glsWeights), searchMethod(DOUBLE_ROUND) { this->problem = _problem; this->searchProblem = _searchProblem; }
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem, std::shared_ptr<GlsWeightSet> _glsWeights, DoubleResolutionMethod _searchMethod) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem, _glsWeights), searchMethod(_searchMethod) { this->problem = _problem; this->searchProblem = _searchProblem; }
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		void updateMapsLength(int pixelWidth);
		void updateMapsDimensions(int pixelWidth, int pixelHeight);

	private:
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMapRounded(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMapDistributed(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMapWeighted(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMapSpacedSingle(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution);
		qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution);
		QPoint getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<RasterPackingProblem> searchProblem;
		DoubleResolutionMethod searchMethod;
	};
}

#endif // RASTEROVERLAPEVALUATOR_H