#include "rasterstrippackingsolvergls.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;


void RasterStripPackingSolverGLS::updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<WeightIncrement> solutionOverlapValues;
	qreal maxOValue = 0;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < originalProblem->count(); itemId1++)
	for (int itemId2 = 0; itemId2 < originalProblem->count(); itemId2++) {
		if (itemId1 == itemId2) continue;
		qreal curOValue = getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
			itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2), originalProblem);
		if (curOValue != 0) {
			solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, (qreal)curOValue));
			if (curOValue > maxOValue) maxOValue = curOValue;
		}
	}

	// Divide vector by maximum
	std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value / maxOValue; });

	// Add to the current weight map
	glsWeights->updateWeights(solutionOverlapValues);
}

//  TODO: Update cache information!
void RasterStripPackingSolverGLS::resetWeights() {
	glsWeights->reset(originalProblem->count());
}

qreal RasterStripPackingSolverGLS::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	qreal totalOverlap = 0;
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += glsWeights->getWeight(itemId, i)*getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i), problem);
	}
	return totalOverlap;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolverGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	return currrentPieceMap;
}