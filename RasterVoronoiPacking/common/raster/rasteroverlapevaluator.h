#ifndef RASTEROVERLAPEVALUATOR_H
#define RASTEROVERLAPEVALUATOR_H

#define ZOOMNEIGHBORHOOD 1

#include "rasterpackingproblem.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

//class MainWindow;

namespace RASTERVORONOIPACKING {

	// --> Class that performs determine overlap operations for solver
	class RasterTotalOverlapMapEvaluator
	{
		friend class ::MainWindow;
	public:
		// --> Default constructor
		RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem);
		RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem, bool cacheMaps);

		// --> Determines the item total overlap map
		virtual QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value);

		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) = 0;
		virtual void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) = 0;
		virtual void resetWeights() = 0;
		virtual std::shared_ptr<GlsWeightSet> getGlsWeights() { return nullptr; }

		// --> Container size update functions
		virtual void updateMapsLength(int pixelWidth);
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight);

	protected:
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

		// --> Pointer to problem
		std::shared_ptr<RasterPackingProblem> problem;

		// --> Set of all item total overlap maps
		TotalOverlapMapSet maps;
	};

	// --> Overlap evaluator class with guided local search metaheuristic
	class RasterTotalOverlapMapEvaluatorGLS : public RasterTotalOverlapMapEvaluator
	{
		friend class ::MainWindow;
	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem) : RasterTotalOverlapMapEvaluator(_problem) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); }
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cacheMaps) : RasterTotalOverlapMapEvaluator(_problem, cacheMaps) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluator(_problem), glsWeights(_glsWeights) {}
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cacheMaps) : RasterTotalOverlapMapEvaluator(_problem, cacheMaps), glsWeights(_glsWeights) {}
		
		// --> Update guided local search weights
		void updateWeights(RasterPackingSolution &solution);
		void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap);
		
		// --> Reset guided local search weights
		void resetWeights();

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);

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
                friend class ::MainWindow;

	public:
		// --> Default constructors
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt) : RasterTotalOverlapMapEvaluatorGLS(_problem), zoomFactorInt(_zoomFactorInt) { createSearchMaps(); }
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt, bool cacheMaps) : RasterTotalOverlapMapEvaluatorGLS(_problem, cacheMaps), zoomFactorInt(_zoomFactorInt) { createSearchMaps(cacheMaps); }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights), zoomFactorInt(_zoomFactorInt) { createSearchMaps(); }
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt, std::shared_ptr<GlsWeightSet> _glsWeights, bool cacheMaps) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, cacheMaps), zoomFactorInt(_zoomFactorInt) { createSearchMaps(cacheMaps); }

		// --> Determines the item total overlap map
		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);
		void updateMapsDimensions(int pixelWidth, int pixelHeight);

	private:
		void createSearchMaps(bool cacheMaps = false);
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution);
		quint32 getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution);
		QPoint getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &val);
		const int zoomFactorInt;
	};
}

#endif // RASTEROVERLAPEVALUATOR_H
