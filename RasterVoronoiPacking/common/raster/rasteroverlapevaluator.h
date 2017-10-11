#ifndef RASTEROVERLAPEVALUATOR_H
#define RASTEROVERLAPEVALUATOR_H

#include "rasterpackingproblem.h"
#include "totaloverlapmap.h"
#include "glsweightset.h"

class MainWindow;

namespace RASTERVORONOIPACKING {

	class RasterOverlapEvaluator
	{
	public:
		RasterOverlapEvaluator(std::shared_ptr<RasterPackingProblem> _problem);
		virtual std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		// --> Guided Local Search functions
		virtual void updateWeights(RasterPackingSolution &solution) = 0;
		virtual void resetWeights() = 0;
		virtual std::shared_ptr<GlsWeightSet> getGlsWeights() { return nullptr; }
		virtual void updateMapsLength(int pixelWidth);
		virtual void updateMapsDimensions(int pixelWidth, int pixelHeight);
		qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		bool detectOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem);
		qreal getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem);
		qreal getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps);

	protected:
		qreal getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2);
		std::shared_ptr<RasterPackingProblem> problem;
		TotalOverlapMapSet maps;
	};

	class RasterOverlapEvaluatorGLS : public RasterOverlapEvaluator
	{
		friend class MainWindow;

	public:
		RasterOverlapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem) : RasterOverlapEvaluator(_problem) { glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); }
		RasterOverlapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterOverlapEvaluator(_problem), glsWeights(_glsWeights) {}

		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		//void getTotalOverlapMap(std::shared_ptr<TotalOverlapMap> currrentPieceMap, int itemId, int orientation, RasterPackingSolution &solution);
		void updateWeights(RasterPackingSolution &solution);
		void resetWeights();

	protected:
		void setgetGlsWeights(std::shared_ptr<GlsWeightSet> _glsWeights) { this->glsWeights = _glsWeights; }
		std::shared_ptr<GlsWeightSet> getGlsWeights() { return glsWeights; }
		std::shared_ptr<GlsWeightSet> glsWeights;
	};

	class RasterOverlapEvaluatorDoubleGLS : public RasterOverlapEvaluatorGLS
	{
		friend class MainWindow;

	public:
		RasterOverlapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem) : RasterOverlapEvaluatorGLS(_searchProblem) { this->problem = _problem; this->searchProblem = _searchProblem; }
		RasterOverlapEvaluatorDoubleGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterPackingProblem> _searchProblem, std::shared_ptr<GlsWeightSet> _glsWeights) : RasterOverlapEvaluatorGLS(_searchProblem, _glsWeights) { this->problem = _problem; this->searchProblem = _searchProblem; }
		std::shared_ptr<TotalOverlapMap> getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution);
		void updateMapsLength(int pixelWidth);
		void updateMapsDimensions(int pixelWidth, int pixelHeight);

	private:
		std::shared_ptr<TotalOverlapMap> getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<TotalOverlapMap> getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution);
		qreal getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution);
		QPoint getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution);
		std::shared_ptr<RasterPackingProblem> searchProblem;
	};
}

#endif // RASTEROVERLAPEVALUATOR_H