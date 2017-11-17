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
			// Check if points are on grid
			int newWidth, newHeight;
			QPoint newReferencePoint;
			if ((curIfp->width() - 1) % zoomFactorInt == 0 && (curIfp->height() - 1) % zoomFactorInt == 0 && curIfp->getOrigin().x() % zoomFactorInt == 0 && (curIfp->height() - 1) % zoomFactorInt == 0) {
				newWidth = 1 + (curIfp->width() - 1) / zoomFactorInt; newHeight = 1 + (curIfp->height() - 1) / zoomFactorInt;
				newReferencePoint = QPoint(curIfp->getOrigin().x() / zoomFactorInt, curIfp->getOrigin().y() / zoomFactorInt);
			}
			else {
				if (curIfp->getOrigin().x() % zoomFactorInt == 0) newReferencePoint.setX(curIfp->getOrigin().x() / zoomFactorInt);
				else newReferencePoint.setX(qFloor((qreal)curIfp->getOrigin().x() / (qreal)zoomFactorInt));
				if (curIfp->getOrigin().y() % zoomFactorInt == 0) newReferencePoint.setY(curIfp->getOrigin().y() / zoomFactorInt);
				else newReferencePoint.setY(qFloor((qreal)curIfp->getOrigin().y() / (qreal)zoomFactorInt));
				int right = -curIfp->getOrigin().x() + curIfp->width() - 1; newWidth = right + newReferencePoint.x() * zoomFactorInt;
				if (newWidth % zoomFactorInt == 0) newWidth = newWidth / zoomFactorInt;
				else newWidth = qFloor((qreal)newWidth / (qreal)zoomFactorInt);
				int top = -curIfp->getOrigin().y() + curIfp->height() - 1; newHeight = top + newReferencePoint.y() * zoomFactorInt;
				if (newHeight % zoomFactorInt == 0) newHeight = newHeight / zoomFactorInt;
				else newHeight = qFloor((qreal)newHeight / (qreal)zoomFactorInt);
				newWidth = newWidth + 1;  newHeight = newHeight + 1;
			}
			std::shared_ptr<TotalOverlapMap> curMap = cacheMaps ?
				std::shared_ptr<TotalOverlapMap>(new CachedTotalOverlapMap(newWidth, newHeight, newReferencePoint, this->problem->count())) :
				std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(newWidth, newHeight, newReferencePoint));
			maps.addOverlapMap(itemId, angle, curMap);

			std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new CachedTotalRectOverlapMap(ZOOMNEIGHBORHOOD*zoomFactorInt, ZOOMNEIGHBORHOOD*zoomFactorInt, QPoint(0, 0), this->problem->count()));
			zoomedMaps.addOverlapMap(itemId, angle, curZoomedMap);
			// FIXME: Delete innerift polygons as they are used to release memomry
		}
	}
}

QPoint RasterTotalOverlapMapEvaluatorDoubleGLS::getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value) {
	bool border;
	QPoint pos = getMinimumOverlapSearchPosition(itemId, orientation, solution, value, border);
	if (value == 0 && !border) return pos;
	int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
	std::shared_ptr<TotalOverlapMap> map = getRectTotalOverlapMap(itemId, orientation, pos, zoomSquareSize, zoomSquareSize, solution);
	QPoint minRelativePos;
	value = map->getMinimum(minRelativePos);
	return minRelativePos;
}

// --> Get absolute minimum overlap position
QPoint RasterTotalOverlapMapEvaluatorDoubleGLS::getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &val, bool &border) {
	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapSearchMap(itemId, orientation, solution);
	QPoint minRelativePos;
	val = map->getMinimum(minRelativePos);
	if (minRelativePos.x() == map->getWidth() - map->getReferencePoint().x() - 1) border = true; else border = false; // FIXME: Does not work in 2D case!

	// Rescale position before returning
	return (int) zoomFactorInt * minRelativePos;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), glsWeights->getWeight(itemId, i), zoomFactorInt);
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
	//std::shared_ptr<CachedTotalRectOverlapMap> curZoomedMap = std::dynamic_pointer_cast<CachedTotalRectOverlapMap>(zoomedMaps.getOverlapMap(itemId, orientation)); curZoomedMap->setRectangle(zoomSquareRect);
	std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(zoomSquareRect));
	
	#ifndef INDIVIDUAL_RECT
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		curZoomedMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	#else
	for (int j = zoomSquareRect.top(); j <= zoomSquareRect.bottom(); j++)
	for (int i = zoomSquareRect.left(); i <= zoomSquareRect.right(); i++)
		curZoomedMap->setValue(QPoint(i, j), getTotalOverlapMapSingleValue(itemId, orientation, QPoint(i, j), solution));
	#endif

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