#include "rasteroverlapevaluator.h"
#include "totaloverlapmap.h"
#include "totaloverlapmapcache.h"

using namespace RASTERVORONOIPACKING;

RasterTotalOverlapMapEvaluatorGLS::RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, bool cache, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem, cuttingStock), maps(_problem->count()) {
	populateMaps(cache, cuttingStock);
}

RasterTotalOverlapMapEvaluatorGLS::RasterTotalOverlapMapEvaluatorGLS(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<GlsWeightSet> _glsWeights, bool cache, bool cuttingStock) : RasterTotalOverlapMapEvaluator(_problem, _glsWeights, cuttingStock), maps(_problem->count()) {
	populateMaps(cache, cuttingStock);
}

void RasterTotalOverlapMapEvaluatorGLS::populateMaps(bool cache, bool cuttingStock) {
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = cache ? std::shared_ptr<TotalOverlapMap>(new CachedTotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle), cuttingStock ? problem->getContainerWidth() : -1, this->problem->count())) :
				std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle), cuttingStock ? problem->getContainerWidth() : -1));
			maps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), getWeight(itemId, i));
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
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), getWeight(itemId, i));
	}
	currrentPieceMap->changeTotalItems(problem->count()); // FIXME: Better way to deal with cached maps
	return currrentPieceMap;
}

// Change cached item total overlap maps lengths according to container size
// pixelWidth is the current container size given in grid size
void RasterTotalOverlapMapEvaluatorGLS::updateMapsLength(int pixelWidth) {
	int deltaPixel = problem->getContainerWidth() - pixelWidth;
	maps.setShrinkVal(deltaPixel);
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
			curMap->setRelativeWidth(deltaPixel);
		}
}

// Change cached item total overlap maps sizes according to container size
// pixelWidth is the current container size given in grid size
void RasterTotalOverlapMapEvaluatorGLS::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	int deltaPixelsX = problem->getContainerWidth() - pixelWidth;
	int deltaPixelsY = problem->getContainerHeight() - pixelHeight;
	for (int itemId = 0; itemId < problem->count(); itemId++)
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
			curMap->setRelativeDimensions(deltaPixelsX, deltaPixelsY);
		}
}