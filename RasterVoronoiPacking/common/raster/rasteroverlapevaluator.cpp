#include "rasteroverlapevaluator.h"

using namespace RASTERVORONOIPACKING;

// --> Get nfp distance value: pos1 is static item and pos2 is orbiting item
qreal getNfpValue(QPoint pos1, QPoint pos2, std::shared_ptr<RasterNoFitPolygon> curNfp, bool &isZero) {
	isZero = false;
	QPoint relPos = pos2 - pos1 + curNfp->getOrigin();

	//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) {
	if (relPos.x() < 0 || relPos.x() > curNfp->width() - 1 || relPos.y() < 0 || relPos.y() > curNfp->height() - 1) {
		isZero = true;
		return 0.0;
	}

	//    int indexValue = curNfp->getImage().pixelIndex(relPos);
	int indexValue = curNfp->getPixel(relPos.x(), relPos.y());
	if (indexValue == 0) {
		isZero = true;
		return 0.0;
	}
	return 1.0 + (curNfp->getMaxD() - 1.0)*((qreal)indexValue - 1.0) / 254.0;
}

RasterOverlapEvaluator::RasterOverlapEvaluator(std::shared_ptr<RasterPackingProblem> _problem) {
	this->problem = _problem;
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<TotalOverlapMap> curMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(problem->getIfps()->getRasterNoFitPolygon(-1, -1, problem->getItemType(itemId), angle)));
			maps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

std::shared_ptr<TotalOverlapMap> RasterOverlapEvaluator::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation), solution.getPosition(i));
	}
	return currrentPieceMap;
}

// --> Change container size
void RasterOverlapEvaluator::updateMapsLength(int pixelWidth) {
	int deltaPixel = problem->getContainerWidth() - pixelWidth;
	maps.setShrinkVal(deltaPixel);
	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->setRelativeWidth(deltaPixel);
	}
}

void RasterOverlapEvaluator::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	int deltaPixelsX = problem->getContainerWidth() - pixelWidth;
	int deltaPixelsY = problem->getContainerHeight() - pixelHeight;
	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->setRelativeDimensions(deltaPixelsX, deltaPixelsY);
	}
}

// --> Get two items minimum overlap
qreal RasterOverlapEvaluator::getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2) {
	std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
	qreal value1Static2Orbiting, value2Static1Orbiting;
	bool feasible;

	curNfp1Static2Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId1), orientation1,
		problem->getItemType(itemId2), orientation2);
	value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
	if (feasible) return 0.0;

	curNfp2Static1Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId2), orientation2,
		problem->getItemType(itemId1), orientation1);
	value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
	if (feasible) return 0.0;

	return value1Static2Orbiting < value2Static1Orbiting ? value1Static2Orbiting : value2Static1Orbiting;
}

qreal RasterOverlapEvaluator::getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem) {
	std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
	qreal value1Static2Orbiting, value2Static1Orbiting;
	bool feasible;

	curNfp1Static2Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId1), orientation1,
		problem->getItemType(itemId2), orientation2);
	value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
	if (feasible) return 0.0;

	curNfp2Static1Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId2), orientation2,
		problem->getItemType(itemId1), orientation1);
	value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
	if (feasible) return 0.0;

	return value1Static2Orbiting < value2Static1Orbiting ? value1Static2Orbiting : value2Static1Orbiting;
}

bool RasterOverlapEvaluator::detectOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem) {
	std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
	qreal value1Static2Orbiting, value2Static1Orbiting;
	bool feasible;

	curNfp1Static2Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId1), orientation1,
		problem->getItemType(itemId2), orientation2);
	value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
	if (feasible) return false;

	curNfp2Static1Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		problem->getItemType(itemId2), orientation2,
		problem->getItemType(itemId1), orientation1);
	value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
	if (feasible) return false;

	if (qFuzzyCompare(1.0 + value1Static2Orbiting, 1.0) || qFuzzyCompare(1.0 + value2Static1Orbiting, 1.0))
		return false;
	return true;
}

qreal RasterOverlapEvaluator::getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	qreal totalOverlap = 0;
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId),
			i, solution.getPosition(i), solution.getOrientation(i), problem);
	}
	return totalOverlap;
}

qreal RasterOverlapEvaluator::getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps) {
	qreal totalOverlap = 0;
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		qreal itemOverlap = getItemTotalOverlap(itemId, solution, problem);
		individualOverlaps.append(itemOverlap);
		totalOverlap += itemOverlap;
	}
	return totalOverlap;
}