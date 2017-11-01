#include "rasteroverlapevaluator.h"
#include "totaloverlapmapcache.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
	newSolution = RasterPackingSolution(originalSolution.getNumItems());
	for (int i = 0; i < originalSolution.getNumItems(); i++) {
		newSolution.setOrientation(i, originalSolution.getOrientation(i));
		QPoint finePos = QPoint(qRound((qreal)originalSolution.getPosition(i).x() * scaleFactor), qRound((qreal)originalSolution.getPosition(i).y() * scaleFactor));
		newSolution.setPosition(i, finePos);
	}
}

void RasterTotalOverlapMapEvaluatorDoubleGLS::createSearchMaps(bool cacheMaps) {
	maps.clear();
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
			std::shared_ptr<RasterNoFitPolygon> curIfp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
			int newWidth = 1 + (curIfp->width() - 1) / zoomFactorInt; int newHeight = 1 + (curIfp->height() - 1) / zoomFactorInt;
			QPoint newReferencePoint = QPoint(curIfp->getOrigin().x() / zoomFactorInt, curIfp->getOrigin().y() / zoomFactorInt);
			std::shared_ptr<TotalOverlapMap> curMap = cacheMaps ?
				std::shared_ptr<TotalOverlapMap>(new CachedTotalOverlapMap(newWidth, newHeight, newReferencePoint, this->problem->count())) :
				std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(newWidth, newHeight, newReferencePoint));
			maps.addOverlapMap(itemId, angle, curMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	QPoint pos = getMinimumOverlapSearchPosition(itemId, orientation, solution);
	int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
	return getRectTotalOverlapMap(itemId, orientation, pos, zoomSquareSize, zoomSquareSize, solution);
}

// --> Get absolute minimum overlap position
QPoint RasterTotalOverlapMapEvaluatorDoubleGLS::getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapSearchMap(itemId, orientation, solution);
	QPoint minRelativePos;
	map->getMinimum(minRelativePos);

	// Rescale position before returning
	return (int) zoomFactorInt * minRelativePos;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i), zoomFactorInt);
	}
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution) {
	// Determine zoomed area inside the innerfit polygon
	std::shared_ptr<RasterNoFitPolygon> curIfp = this->problem->getIfps()->getRasterNoFitPolygon(0, 0, this->problem->getItemType(itemId), orientation);
	QRect curIfpBoundingBox(QPoint(-curIfp->getOriginX(), -curIfp->getOriginY()), QSize(curIfp->width() - maps.getShrinkValX(), curIfp->height() - maps.getShrinkValY()));
	QRect zoomSquareRect(QPoint(pos.x() - width / 2, pos.y() - height / 2), QSize(width, height));
	zoomSquareRect = zoomSquareRect.intersected(curIfpBoundingBox);

	// Create zoomed overlap Map. FIXME: Use cache map?
	std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(zoomSquareRect));
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		curZoomedMap->addVoronoi(i, problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}

	return curZoomedMap;
}

int getRoughShrinkage(int deltaPixels, int zoomFactorInt) {
	qreal fracDeltaPixelsX = (qreal)(deltaPixels) / (qreal)zoomFactorInt;
	if (qFuzzyCompare(1.0 + fracDeltaPixelsX, 1.0 + qRound(fracDeltaPixelsX))) return qRound(fracDeltaPixelsX);
	return qRound(fracDeltaPixelsX);
}

void RasterTotalOverlapMapEvaluatorDoubleGLS::updateMapsLength(int pixelWidth) {
	int deltaPixels = problem->getContainerWidth() - pixelWidth;
	maps.setShrinkVal(deltaPixels);
	int deltaPixelsRough = getRoughShrinkage(deltaPixels, zoomFactorInt);

	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->setRelativeWidth(deltaPixelsRough);
	}
}

void RasterTotalOverlapMapEvaluatorDoubleGLS::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	int deltaPixelsX = problem->getContainerWidth() - pixelWidth;
	int deltaPixelsY = problem->getContainerHeight() - pixelHeight;
	maps.setShrinkVal(deltaPixelsX, deltaPixelsY);
	int deltaPixelsRoughX = getRoughShrinkage(deltaPixelsX, zoomFactorInt);
	int deltaPixelsRoughY = getRoughShrinkage(deltaPixelsY, zoomFactorInt);

	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->setRelativeDimensions(deltaPixelsRoughX, deltaPixelsRoughY);
	}
}

quint32 RasterTotalOverlapMapEvaluatorDoubleGLS::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution) {
	quint32 totalOverlap = 0;
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += glsWeights->getWeight(itemId, i) * problem->getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i));
	}
	return totalOverlap;
}