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
		RasterTotalOverlapMapEvaluator() {};
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
		RasterTotalOverlapMapEvaluatorGLS() : RasterTotalOverlapMapEvaluator() {}
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluator(), glsWeights(_glsWeights) {}
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
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem), searchProblemScale(_searchProblem->getScale()) { this->problem = _problem; this->searchProblem = _searchProblem; }
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluatorGLS(_searchProblem, _glsWeights), searchProblemScale(_searchProblem->getScale()) { this->problem = _problem; this->searchProblem = _searchProblem; }
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		void updateMapsLength(int pixelWidth);
		void updateMapsDimensions(int pixelWidth, int pixelHeight);

	protected:
		RasterTotalOverlapMapEvaluatorDoubleGLS(int _searchProblemScale) : RasterTotalOverlapMapEvaluatorGLS(), searchProblemScale(_searchProblemScale) {}
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<GlsWeightSet> _glsWeights, int _searchProblemScale) : RasterTotalOverlapMapEvaluatorGLS(_glsWeights), searchProblemScale(_searchProblemScale) {}
		const int searchProblemScale;

	private:
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution);
		qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution);
		QPoint getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<RasterPackingProblem> searchProblem;
	};

	class RasterTotalOverlapMapEvaluatorDoubleGLSSingle : public RasterTotalOverlapMapEvaluatorDoubleGLS
	{
	public:
		RasterTotalOverlapMapEvaluatorDoubleGLSSingle(std::shared_ptr<RasterPackingProblem> _problem, int zoomFactorInt);
		RasterTotalOverlapMapEvaluatorDoubleGLSSingle(std::shared_ptr<RasterPackingProblem> _problem, int zoomFactorInt, std::shared_ptr<GlsWeightSet> _glsWeights);

	private:
		void createSearchMaps();
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution);
		
	};
}

#endif // RASTEROVERLAPEVALUATOR_H