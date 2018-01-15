#ifndef RASTERRECTPACKINGCOMPACTOR_H
#define RASTERRECTPACKINGCOMPACTOR_H

#include "rastersquarepackingcompactor.h"

namespace RASTERVORONOIPACKING {
	class RasterRectangularPackingCompactor : public RasterSquarePackingCompactor
	{
		friend class ::MainWindow;
	public:
		RasterRectangularPackingCompactor(int initialLength, int initialHeight, std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterSquarePackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc) {
			if (initialLength > 0 && initialHeight > 0) {
				QPair<int, int> newDimensions = setContainerDimensions(initialLength, initialHeight);
				curRealLength = newDimensions.first; curRealHeight = newDimensions.second;
			}
			bestArea = std::numeric_limits<qreal>::max();
		}
		RasterRectangularPackingCompactor(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterSquarePackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc) {
			bestArea = std::numeric_limits<qreal>::max();
		}

		void generateRandomSolution(RasterPackingSolution &solution);
		void generateBottomLeftSolution(RasterPackingSolution &solution);
		virtual bool shrinkContainer(RasterPackingSolution &solution) = 0;
		virtual bool expandContainer(RasterPackingSolution &solution) = 0;
		int getCurrentLength() { return qRound(curRealLength); }
		int getCurrentHeight() { return qRound(curRealHeight); }
		ProblemType getProblemType() { return RectangularPacking; }

	protected:
		QPair<int, int> setContainerDimensions(int newLength, int newHeight);
		bool getShrinkedDimension(qreal realDim, qreal &newRealDim, int minimumDimension);
		qreal curRealLength, curRealHeight;
		qreal bestArea;

	private:
		void setContainerDimensions(int newLength, int newHeight, RasterPackingSolution &solution);// DEBUG ONLY
	};

	class RasterRectangularPackingRandomCompactor : public RasterRectangularPackingCompactor
	{
	public:
		RasterRectangularPackingRandomCompactor(int initialLength, int initialHeight, std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterRectangularPackingCompactor(initialLength, initialHeight, _problem, _overlapEvaluator, _rdec, _rinc) {}
		RasterRectangularPackingRandomCompactor(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterRectangularPackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc)  {}

		bool shrinkContainer(RasterPackingSolution &solution);
		bool expandContainer(RasterPackingSolution &solution);

	private:
		void randomShrinkDimensions(bool changeLength, qreal ratio);
	};

	class RasterRectangularPackingBagpipeCompactor : public RasterRectangularPackingCompactor {
	public:
		RasterRectangularPackingBagpipeCompactor(int initialLength, int initialHeight, std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterRectangularPackingCompactor(initialLength, initialHeight, _problem, _overlapEvaluator, _rdec, _rinc), bagpipeDirection(1) {}
		RasterRectangularPackingBagpipeCompactor(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterRectangularPackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc), bagpipeDirection(1) {}
		bool shrinkContainer(RasterPackingSolution &solution);
		bool expandContainer(RasterPackingSolution &solution);

	private:
		bool bagpipeDirection;
	};
}

#endif // RASTERRECTPACKINGCOMPACTOR_H