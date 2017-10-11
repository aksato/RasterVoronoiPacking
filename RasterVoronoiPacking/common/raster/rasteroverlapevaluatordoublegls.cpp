#include "rasteroverlapevaluator.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

#define ZOOMNEIGHBORHOOD 3

void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
	newSolution = RasterPackingSolution(originalSolution.getNumItems());
	for (int i = 0; i < originalSolution.getNumItems(); i++) {
		newSolution.setOrientation(i, originalSolution.getOrientation(i));
		QPoint finePos = QPoint(qRound((qreal)originalSolution.getPosition(i).x() * scaleFactor), qRound((qreal)originalSolution.getPosition(i).y() * scaleFactor));
		newSolution.setPosition(i, finePos);
	}
}

std::shared_ptr<TotalOverlapMap> RasterOverlapEvaluatorDoubleGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution) {
	QPoint pos = getMinimumOverlapSearchPosition(itemId, orientation, solution);
	int zoomSquareSize = ZOOMNEIGHBORHOOD*qRound(this->problem->getScale() / this->searchProblem->getScale());
	return getRectTotalOverlapMap(itemId, orientation, pos, zoomSquareSize, zoomSquareSize, solution);
}

// --> Get absolute minimum overlap position
QPoint RasterOverlapEvaluatorDoubleGLS::getMinimumOverlapSearchPosition(int itemId, int orientation, RasterPackingSolution &solution) {
	// Scale solution to seach scale
	RasterPackingSolution roughSolution;
	qreal zoomFactor = this->problem->getScale() / this->searchProblem->getScale();
	getScaledSolution(solution, roughSolution, 1.0 / zoomFactor);

	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapSearchMap(itemId, orientation, roughSolution);
	//float fvalue = value;
	float fvalue;
	QPoint minRelativePos = map->getMinimum(fvalue);

	// Rescale position before returning
	return zoomFactor*(minRelativePos - map->getReferencePoint());
}

std::shared_ptr<TotalOverlapMap> RasterOverlapEvaluatorDoubleGLS::getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < searchProblem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(searchProblem->getNfps()->getRasterNoFitPolygon(searchProblem->getItemType(i), solution.getOrientation(i), searchProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	return currrentPieceMap;
}

std::shared_ptr<TotalOverlapMap> RasterOverlapEvaluatorDoubleGLS::getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution) {
	// Determine zoomed area inside the innerfit polygon
	std::shared_ptr<RasterNoFitPolygon> curIfp = this->problem->getIfps()->getRasterNoFitPolygon(-1, -1, this->problem->getItemType(itemId), orientation);
	QRect curIfpBoundingBox(QPoint(-curIfp->getOriginX(), -curIfp->getOriginY()), QSize(curIfp->width() - maps.getShrinkVal(), curIfp->height()));
	QRect zoomSquareRect(QPoint(pos.x() - width / 2, pos.y() - height / 2), QSize(width, height));
	zoomSquareRect = zoomSquareRect.intersected(curIfpBoundingBox);

	// Create zoomed overlap Map
	std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(zoomSquareRect));
	for (int j = zoomSquareRect.top(); j <= zoomSquareRect.bottom(); j++)
	for (int i = zoomSquareRect.left(); i <= zoomSquareRect.right(); i++)
		curZoomedMap->setValue(QPoint(i, j), getTotalOverlapMapSingleValue(itemId, orientation, QPoint(i, j), solution));

	return curZoomedMap;
}

void RasterOverlapEvaluatorDoubleGLS::updateMapsLength(int pixelWidth) {
	int deltaPixel = problem->getContainerWidth() - pixelWidth;
	maps.setShrinkVal(deltaPixel);

	//int deltaPixelsRough;
	//qreal sizeRatio = this->searchProblem->getScale() / this->problem->getScale();
	//qreal fracDeltaPixelsX = sizeRatio*(qreal)(problem->getContainerWidth() - pixelWidth);
	//if (qFuzzyCompare(1.0 + fracDeltaPixelsX, 1.0 + qRound(fracDeltaPixelsX))) deltaPixelsRough = qRound(fracDeltaPixelsX);
	//else deltaPixelsRough = qFloor(fracDeltaPixelsX);

	for (int itemId = 0; itemId < problem->count(); itemId++)
	for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		//curMap->setRelativeWidth(deltaPixelsRough);
		std::shared_ptr<RasterNoFitPolygon> curIfp = this->problem->getIfps()->getRasterNoFitPolygon(-1, -1, this->problem->getItemType(itemId), angle);
		int leftMostPointX = curIfp->width() - curIfp->getOriginX() - 1 - deltaPixel;
		int newLeftMostPointX = qCeil(leftMostPointX*this->searchProblem->getScale() / this->problem->getScale());
		int newTotalOverlapMapWidth = newLeftMostPointX + curMap->getReferencePoint().x() + 1;
		curMap->setWidth(newTotalOverlapMapWidth);
	}
}

void RasterOverlapEvaluatorDoubleGLS::updateMapsDimensions(int pixelWidth, int pixelHeight) {
	// TODO: Implement 2D double GLS

	//int deltaPixelsX, deltaPixelsY;
	//qreal sizeRatio = this->searchProblem->getScale() / this->problem->getScale();

	//qreal fracDeltaPixelsX = sizeRatio*(qreal)(problem->getContainerWidth() - pixelWidth);
	//if (qFuzzyCompare(1.0 + fracDeltaPixelsX, 1.0 + qRound(fracDeltaPixelsX))) deltaPixelsX = qRound(fracDeltaPixelsX);
	//else deltaPixelsX = qFloor(fracDeltaPixelsX);

	////int deltaPixelsY = qRound(sizeRatio*(problem->getContainerHeight() - pixelHeight));
	//for (int itemId = 0; itemId < problem->count(); itemId++)
	//for (uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
	//	std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
	//	curMap->setRelativeDimensions(deltaPixelsX, deltaPixelsY);
	//}
}

qreal RasterOverlapEvaluatorDoubleGLS::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution) {
	qreal totalOverlap = 0;
	for (int i = 0; i < problem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += glsWeights->getWeight(itemId, i) * getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i));
	}
	return totalOverlap;
}
