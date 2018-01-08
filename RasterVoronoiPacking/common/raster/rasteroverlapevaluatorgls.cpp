#include "rasteroverlapevaluator.h"

using namespace RASTERVORONOIPACKING;

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorGLS::getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	currrentPieceMap->changeTotalItems(placedItems.length()+1); // FIXME: Better way to deal with cached maps
	for (int i : placedItems) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	currrentPieceMap->changeTotalItems(problem->count()); // FIXME: Better way to deal with cached maps
	return currrentPieceMap;
}

void RasterTotalOverlapMapEvaluatorGLS::updateWeights(RasterPackingSolution &solution) {
	QVector<WeightIncrement> solutionOverlapValues;
	//quint32 maxOValue = 0;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < problem->count(); itemId1++)
	for (int itemId2 = 0; itemId2 < problem->count(); itemId2++) {
		if (itemId1 == itemId2) continue;
		quint32 curOValue = problem->getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
			itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2));
		if (curOValue != 0) {
			solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, 1));
			//if (curOValue > maxOValue) maxOValue = curOValue;
		}
	}

	// FIXME: Integer approx.
	//// Divide vector by maximum
	//std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value / maxOValue; });

	// Add to the current weight map
	glsWeights->updateWeights(solutionOverlapValues);
}

void RasterTotalOverlapMapEvaluatorGLS::updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) {
	std::transform(glsWeights->begin(), glsWeights->end(), overlaps.begin(),
		glsWeights->begin(), [&maxOverlap](const quint32 &a, const quint32 &b){return a + qRound(100.0*(qreal)b / (qreal)maxOverlap); });
}

//  TODO: Update cache information!
void RasterTotalOverlapMapEvaluatorGLS::resetWeights() {
	glsWeights->reset(problem->count());
}