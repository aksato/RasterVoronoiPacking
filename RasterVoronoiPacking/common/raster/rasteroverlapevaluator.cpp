#include "rasteroverlapevaluator.h"
#include "totaloverlapmapcache.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluator::RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem, bool cuttingStock) : problem (_problem) {
	glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count()));
}

RasterTotalOverlapMapEvaluator::RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cuttingStock) : problem(_problem), glsWeights(_glsWeights) {
}

QPoint RasterTotalOverlapMapEvaluator::getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value) {
	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapMap(itemId, orientation, solution);
	QPoint minRelativePos;
	value = map->getMinimum(minRelativePos);
	return minRelativePos;
}

QPoint RasterTotalOverlapMapEvaluator::getBottomLeftPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	std::shared_ptr<TotalOverlapMap> map = getPartialTotalOverlapMap(itemId, orientation, solution, placedItems);
	QPoint minRelativePos;
	map->getBottomLeft(minRelativePos);
	return minRelativePos;
}

void RasterTotalOverlapMapEvaluator::updateWeights(RasterPackingSolution &solution) {
	QVector<WeightIncrement> solutionOverlapValues;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < problem->count(); itemId1++)
		for (int itemId2 = 0; itemId2 < problem->count(); itemId2++) {
			if (itemId1 == itemId2) continue;
			quint32 curOValue = problem->getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
				itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2));
			if (curOValue != 0) {
				solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, 1));
			}
		}

	// Add to the current weight map
	glsWeights->updateWeights(solutionOverlapValues);
}

void RasterTotalOverlapMapEvaluator::updateWeights(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 maxOverlap) {
	std::transform(glsWeights->begin(), glsWeights->end(), overlaps.begin(),
		glsWeights->begin(), [&maxOverlap](const quint32 &a, const quint32 &b) {return a + qRound(100.0*(qreal)b / (qreal)maxOverlap); });
}

//  TODO: Update cache information!
void RasterTotalOverlapMapEvaluator::resetWeights() {
	glsWeights->reset(problem->count());
}

// Determines the item total overlap map for a given orientation in a solution
std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorNoGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		//currrentPieceMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation), solution.getPosition(i));
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i));
	}
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorNoGLS::getPartialTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	currrentPieceMap->changeTotalItems(placedItems.length()+1); // FIXME: Better way to deal with cached maps
	for (int i : placedItems) {
		if (i == itemId) continue;
		//currrentPieceMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation), solution.getPosition(i));
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i));
	}
	currrentPieceMap->changeTotalItems(problem->count()); // FIXME: Better way to deal with cached maps
	return currrentPieceMap;
}