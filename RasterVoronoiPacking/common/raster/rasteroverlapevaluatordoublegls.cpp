#include "rasteroverlapevaluator.h"
#include "totaloverlapmapcache.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

int floorDivide(int coord, int scale) {
	if (coord % scale == 0) return coord / scale;
	return qFloor((qreal)coord / (qreal)scale);
}

int ceilDivide(int coord, int scale) {
	if (coord % scale == 0) return coord / scale;
	return qCeil((qreal)coord / (qreal)scale);
}

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
			if ((curIfp->width() - 1) % zoomFactorInt == 0 && (curIfp->height() - 1) % zoomFactorInt == 0 && curIfp->getOrigin().x() % zoomFactorInt == 0 && (curIfp->height() - 1) % zoomFactorInt == 0 && curIfp->getOrigin().y() % zoomFactorInt == 0) {
				newWidth = 1 + (curIfp->width() - 1) / zoomFactorInt; newHeight = 1 + (curIfp->height() - 1) / zoomFactorInt;
				newReferencePoint = QPoint(curIfp->getOrigin().x() / zoomFactorInt, curIfp->getOrigin().y() / zoomFactorInt);
			}
			else {
				int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
				newReferencePoint.setX(floorDivide(curIfp->getOrigin().x() + zoomSquareSize / 2, zoomFactorInt));
				newReferencePoint.setY(floorDivide(curIfp->getOrigin().y() + zoomSquareSize / 2, zoomFactorInt));
				int right = ceilDivide(-curIfp->getOrigin().x() + curIfp->width() - 1 - zoomSquareSize / 2, zoomFactorInt);
				int top = ceilDivide(-curIfp->getOrigin().y() + curIfp->height() - 1 - zoomSquareSize / 2, zoomFactorInt);
				newWidth = right + newReferencePoint.x() + 1;
				newHeight = top + newReferencePoint.y() + 1;
			}
			std::shared_ptr<TotalOverlapMap> curMap = cacheMaps ?
				std::shared_ptr<TotalOverlapMap>(new CachedTotalOverlapMap(newWidth, newHeight, newReferencePoint, this->problem->count())) :
				std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(newWidth, newHeight, newReferencePoint));
			maps.addOverlapMap(itemId, angle, curMap);
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
	border = false;
	if (minRelativePos.x() == map->getWidth() - map->getReferencePoint().x() - 1 ||
		minRelativePos.y() == map->getHeight() - map->getReferencePoint().y() - 1 || 
		minRelativePos.x() == - map->getReferencePoint().x() || 
		minRelativePos.y() == - map->getReferencePoint().y()) 
		border = true; // FIXME: Does not work in 2D case!

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
	QRect zoomSquareRect(QPoint(pos.x() - width / 2, pos.y() - height / 2), QPoint(pos.x() + width / 2, pos.y() + height / 2));
	zoomSquareRect = zoomSquareRect.intersected(curIfpBoundingBox);

	// Create zoomed overlap Map. FIXME: Use cache map?
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

	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		// Determine new width
		std::shared_ptr<RasterNoFitPolygon> curIfp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
		int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
		int curWidth = curIfp->width() - deltaPixels;
		int right = ceilDivide(-curIfp->getOrigin().x() + curWidth - 1 - zoomSquareSize / 2, zoomFactorInt);
		int newWidth = right + curMap->getReferencePoint().x() + 1;
		curMap->setRelativeWidth(curMap->getOriginalWidth() - newWidth);
	}
}

void RasterTotalOverlapMapEvaluatorDoubleGLS::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	int deltaPixelsX = problem->getContainerWidth() - pixelWidth;
	int deltaPixelsY = problem->getContainerHeight() - pixelHeight;
	maps.setShrinkVal(deltaPixelsX, deltaPixelsY);

	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		// Determine new dimensions
		std::shared_ptr<RasterNoFitPolygon> curIfp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), angle);
		int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
		int curWidth  = curIfp->width()  - deltaPixelsX; int right = ceilDivide(-curIfp->getOrigin().x() + curWidth  - 1 - zoomSquareSize / 2, zoomFactorInt); int newWidth  = right + curMap->getReferencePoint().x() + 1;
		int curHeight = curIfp->height() - deltaPixelsY; int   top = ceilDivide(-curIfp->getOrigin().y() + curHeight - 1 - zoomSquareSize / 2, zoomFactorInt); int newHeight =   top + curMap->getReferencePoint().y() + 1;
		curMap->setRelativeDimensions(curMap->getOriginalWidth() - newWidth, curMap->getOriginalHeight() - newHeight);
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

QPoint RasterTotalOverlapMapEvaluatorDoubleGLS::getBottomLeftPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	QPoint pos = getBottomLeftPartialSearchPosition(itemId, orientation, solution, placedItems);
	int zoomSquareSize = ZOOMNEIGHBORHOOD*zoomFactorInt;
	std::shared_ptr<TotalOverlapMap> map = getPartialRectTotalOverlapMap(itemId, orientation, pos, zoomSquareSize, zoomSquareSize, solution, placedItems);
	QPoint minRelativePos;
	map->getMinimum(minRelativePos);
	return minRelativePos;
}

QPoint RasterTotalOverlapMapEvaluatorDoubleGLS::getBottomLeftPartialSearchPosition(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	std::shared_ptr<TotalOverlapMap> map = getPartialTotalOverlapSearchMap(itemId, orientation, solution, placedItems);
	QPoint minRelativePos;
	map->getBottomLeft(minRelativePos);

	// Rescale position before returning
	return (int)zoomFactorInt * minRelativePos;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getPartialTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution, QList<int> &placedItems) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	currrentPieceMap->changeTotalItems(placedItems.length()+1); // FIXME: Better way to deal with cached maps
	for (int i : placedItems) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(i, curItemNfpSet->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i)), solution.getPosition(i), glsWeights->getWeight(itemId, i), zoomFactorInt);
	}
	currrentPieceMap->changeTotalItems(problem->count()); // FIXME: Better way to deal with cached maps
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterTotalOverlapMapEvaluatorDoubleGLS::getPartialRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, QList<int> &placedItems) {
	// Determine zoomed area inside the innerfit polygon
	std::shared_ptr<RasterNoFitPolygon> curIfp = this->problem->getIfps()->getRasterNoFitPolygon(0, 0, this->problem->getItemType(itemId), orientation);
	QRect curIfpBoundingBox(QPoint(-curIfp->getOriginX(), -curIfp->getOriginY()), QSize(curIfp->width() - maps.getShrinkValX(), curIfp->height() - maps.getShrinkValY()));
	QRect zoomSquareRect(QPoint(pos.x() - width / 2, pos.y() - height / 2), QPoint(pos.x() + width / 2, pos.y() + height / 2));
	zoomSquareRect = zoomSquareRect.intersected(curIfpBoundingBox);

	// Create zoomed overlap Map. FIXME: Use cache map?
	std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(zoomSquareRect));
#ifndef INDIVIDUAL_RECT
	std::shared_ptr<ItemRasterNoFitPolygonSet> curItemNfpSet = problem->getNfps()->getItemRasterNoFitPolygonSet(problem->getItemType(itemId), orientation);
	for (int i : placedItems) {
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