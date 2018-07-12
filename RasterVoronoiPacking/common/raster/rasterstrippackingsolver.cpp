#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

std::shared_ptr<RasterStripPackingSolver> RasterStripPackingSolver::createRasterPackingSolver(std::shared_ptr<RasterPackingProblem> problem, RasterStripPackingParameters &parameters) {
	std::shared_ptr<RasterStripPackingSolver> solver;
	std::shared_ptr<GlsWeightSet> weights;
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator;

	// Determine weight
	if (parameters.getHeuristic() == NONE) weights = std::shared_ptr<GlsNoWeightSet>(new GlsNoWeightSet); // No guided local search
	else weights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); // GLS

	// Determine overlap evaluator
	if (parameters.getZoomFactor() > 1)
		overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorDoubleGLS>(new RasterTotalOverlapMapEvaluatorDoubleGLS(problem, parameters.getZoomFactor(), weights));
	else overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(problem, weights, parameters.getCompaction() == RASTERVORONOIPACKING::CUTTINGSTOCK));
	if (!parameters.isCacheMaps()) overlapEvaluator->disableMapCache();
	// Create solver
	solver = std::shared_ptr<RasterStripPackingSolver>(new RasterStripPackingSolver(problem, overlapEvaluator));
	return solver;
}

RasterStripPackingSolver::RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator) : overlapEvaluator(_overlapEvaluator) {
    this->originalProblem = _problem;
}

qreal getItemMaxDimension(std::shared_ptr<RasterPackingProblem> problem, int itemId) {
	qreal maxDim = 0;
	for (unsigned int angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		qreal curMaxDim = problem->getItem(itemId)->getMaxX(angle) - problem->getItem(itemId)->getMinX(angle);
		if (curMaxDim > maxDim) maxDim = curMaxDim;
	}
	return maxDim;
}

// --> Get layout overlap (sum of individual overlap values)
quint32 RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution) {
	quint32 totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		totalOverlap += getItemTotalOverlap(itemId, solution);
    }
    return totalOverlap;
}

quint32 RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 &maxOverlap) {
	maxOverlap = 0;
	quint32 totalOverlap = 0;
	int overlapIndex = 0;
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		quint32 curOverlap = 0;
		int i = 0;
		for (; i < itemId + 1; i++) overlaps[overlapIndex++] = 0;
		for (; i < originalProblem->count(); i++) {
			quint32 individualOverlap = originalProblem->getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId), i, solution.getPosition(i), solution.getOrientation(i));
			overlaps[overlapIndex++] = individualOverlap;
			curOverlap += individualOverlap;
			if (individualOverlap > maxOverlap) maxOverlap = individualOverlap;
		}
		totalOverlap += 2*curOverlap;
	}
	return totalOverlap;
}

void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	for (int i = 0; i < originalProblem->count(); i++) {
		int shuffledId = sequence[i];
		#ifndef NOSKIPFEASIBLE
		if (!detectItemTotalOverlap(shuffledId, solution)) continue;
		#endif
		quint32 minValue; QPoint minPos; int minAngle = 0;
		minPos = overlapEvaluator->getMinimumOverlapPosition(shuffledId, minAngle, solution, minValue);
		if (minValue != 0) {
			for (uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
				quint32 curValue; QPoint curPos;
				curPos = overlapEvaluator->getMinimumOverlapPosition(shuffledId, curAngle, solution, curValue);
				if (curValue < minValue) {
					minValue = curValue; minPos = curPos; minAngle = curAngle;
					if (minValue == 0) break;
				}
			}
		}
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

void getNextBLPosition(QPoint &curPos, int  minIfpX, int minIfpY, int maxIfpX, int maxIfpY) {
	curPos.setY(curPos.y() + 1); 
	if (curPos.y() > maxIfpY) { 
		curPos.setY(minIfpY); curPos.setX(curPos.x() + 1);
	}
}

quint32 RasterStripPackingSolver::getItemTotalOverlap(int itemId, RasterPackingSolution &solution) {
	quint32 totalOverlap = 0;
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += originalProblem->getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId),
			i, solution.getPosition(i), solution.getOrientation(i));
	}
	return totalOverlap;
}

bool RasterStripPackingSolver::detectItemTotalOverlap(int itemId, RasterPackingSolution &solution) {
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		if (originalProblem->areOverlapping(itemId, solution.getPosition(itemId), solution.getOrientation(itemId), i, solution.getPosition(i), solution.getOrientation(i))) return true;
	}
	return false;
}