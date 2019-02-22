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
		RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false);

		// --> Determines the item total overlap map
		virtual QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value);
		virtual QPoint getBottomLeftPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);

		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) = 0;
		virtual void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) = 0;
		virtual void resetWeights() = 0;
		virtual std::shared_ptr<GlsWeightSet> getGlsWeights() { return nullptr; }

		// --> Container size update functions
		virtual void updateMapsLength(int pixelWidth);
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight);

		// --> Option functions
		virtual void disableMapCache();
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
	protected:
		
		virtual std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);

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
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock = false) : RasterTotalOverlapMapEvaluator(_problem, cuttingStock) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluator(_problem, cuttingStock), glsWeights(_glsWeights) {}
		
		// --> Update guided local search weights
		void updateWeights(RasterPackingSolution &solution);
		void updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap);
		
		// --> Reset guided local search weights
		void resetWeights();

	protected:
		// --> Determines the item total overlap map with guided local search
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		virtual std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);

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
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, cuttingStock), zoomFactorInt(_zoomFactorInt) { createSearchMaps(cuttingStock); }

		// --> Constructors using a custom weights
		RasterTotalOverlapMapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, int _zoomFactorInt, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock = false) : RasterTotalOverlapMapEvaluatorGLS(_problem, _glsWeights, cuttingStock), zoomFactorInt(_zoomFactorInt) { createSearchMaps(cuttingStock); }

		// --> Determines the item total overlap map
		QPoint getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value);
		QPoint getBottomLeftPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);

		// --> Container size update functions
		void updateMapsLength(int pixelWidth);
		void updateMapsDimensions(int pixelWidth, int pixelHeight);

		// --> Option functions
		void disableMapCache();
	private:
		void createSearchMaps(bool cuttingStock);
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, int stockLocation);
		quint32 getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution);
		// --> Get absolute minimum overlap position
		QPoint getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &val, bool &border, int &stockLocation);
		// --> Bottom left functions (partial overlap evaluation)
		QPoint getBottomLeftPartialSearchPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);
		std::shared_ptr<TotalOverlapMap> getPartialTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems);
		std::shared_ptr<TotalOverlapMap> getPartialRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, QList<int> &placedItems);
		// --> Zoom factor
		const int zoomFactorInt;
	};
}

#endif // RASTEROVERLAPEVALUATOR_H
