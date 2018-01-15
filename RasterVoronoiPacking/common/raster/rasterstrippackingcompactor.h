#ifndef RASTERSTRIPPACKINGCOMPACTOR_H
#define RASTERSTRIPPACKINGCOMPACTOR_H

#include "rasteroverlapevaluator.h"
namespace RASTERVORONOIPACKING {
	enum ProblemType { StripPacking, SquarePacking, RectangularPacking };

	class RasterStripPackingCompactor
	{
		friend class ::MainWindow;
	public:
		RasterStripPackingCompactor(int initialWidth, std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : problem(_problem), overlapEvaluator(_overlapEvaluator), rdec(_rdec), rinc(_rinc) {
			if (initialWidth > 0) {
				curRealLength = setContainerWidth(initialWidth);
				bestWidth = std::numeric_limits<int>::max();
			}
		}
		RasterStripPackingCompactor(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : problem(_problem), overlapEvaluator(_overlapEvaluator), rdec(_rdec), rinc(_rinc) {}

		virtual void generateRandomSolution(RasterPackingSolution &solution);
		virtual void generateBottomLeftSolution(RasterPackingSolution &solution);
		virtual bool shrinkContainer(RasterPackingSolution &solution);
		virtual bool expandContainer(RasterPackingSolution &solution);
		virtual int getCurrentLength() { return qRound(curRealLength); }
		virtual int getCurrentHeight() { return problem->getContainerHeight(); }
		virtual ProblemType getProblemType() { return StripPacking; }
	private:
		int setContainerWidth(int newWitdh);
		qreal getItemMaxDimension(int itemId);
		int bestWidth;
		qreal curRealLength;

	protected:
		virtual void setContainerWidth(int newWitdh, RasterPackingSolution &solution);// DEBUG ONLY

		std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator;
		std::shared_ptr<RasterPackingProblem> problem;
		qreal rdec, rinc;
	};

}

#endif // RASTERSTRIPPACKINGCOMPACTOR_H