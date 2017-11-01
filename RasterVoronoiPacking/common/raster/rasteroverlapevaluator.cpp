#include "rasteroverlapevaluator.h"
#include "totaloverlapmapcache.h"

using namespace RASTERVORONOIPACKING;

// Default constructor
RasterTotalOverlapMapEvaluator::RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem) : maps(_problem->count()) {
	this->problem = _problem;
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle)));
			maps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

RasterTotalOverlapMapEvaluator::RasterTotalOverlapMapEvaluator(std::shared_ptr<RasterPackingProblem> _problem, bool cacheMaps) : maps(_problem->count()) {
	this->problem = _problem;
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = cacheMaps ? std::shared_ptr<TotalOverlapMap>(new CachedTotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle), this->problem->count()))
				: std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle)));
			maps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

// Determines the item total overlap map for a given orientation in a solution
std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluator::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
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

// Change cached item total overlap maps lengths according to container size
// pixelWidth is the current container size given in grid size
void RasterTotalOverlapMapEvaluator::updateMapsLength(int pixelWidth) {
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
void RasterTotalOverlapMapEvaluator::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	int deltaPixelsX = problem->getContainerWidth() - pixelWidth;
	int deltaPixelsY = problem->getContainerHeight() - pixelHeight;
	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->setRelativeDimensions(deltaPixelsX, deltaPixelsY);
	}
}