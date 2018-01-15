#ifndef RASTERSQUAREPACKINGCOMPACTOR_H
#define RASTERSQUAREPACKINGCOMPACTOR_H

#include "rasterstrippackingcompactor.h"

namespace RASTERVORONOIPACKING {
	class RasterSquarePackingCompactor : public RasterStripPackingCompactor
	{
	public:
		RasterSquarePackingCompactor(int initialSize, std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterStripPackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc) {
			if (initialSize > 0) {
				curRealSize = setContainerSize(initialSize);
				bestSize = std::numeric_limits<int>::max();
			}
		}
		RasterSquarePackingCompactor(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator, qreal _rdec, qreal _rinc) : RasterStripPackingCompactor(_problem, _overlapEvaluator, _rdec, _rinc) {}

		void generateRandomSolution(RasterPackingSolution &solution);
		void generateBottomLeftSolution(RasterPackingSolution &solution);
		bool shrinkContainer(RasterPackingSolution &solution);
		bool expandContainer(RasterPackingSolution &solution);
		int getCurrentLength() { return qRound(curRealSize); }
		int getCurrentHeight() { return qRound(curRealSize); }
		ProblemType getProblemType() { return SquarePacking; }

	protected:
		void generateBottomLeftLayout(RasterPackingSolution &solution, int &length, int &height);

	private:
		bool detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution);
		int setContainerSize(int newSize);
		qreal curRealSize;
		int bestSize;
	};
}

#endif // RASTERSQUAREPACKINGCOMPACTOR_H