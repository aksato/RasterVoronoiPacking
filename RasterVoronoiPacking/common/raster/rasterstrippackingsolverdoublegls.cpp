#include "rasterstrippackingsolverdoublegls.h"
#include <QtCore/qmath.h>

#define ZOOMNEIGHBORHOOD 3

using namespace RASTERVORONOIPACKING;

void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
	newSolution = RasterPackingSolution(originalSolution.getNumItems());
	for (int i = 0; i < originalSolution.getNumItems(); i++) {
		newSolution.setOrientation(i, originalSolution.getOrientation(i));
		QPoint finePos = QPoint(qRound((qreal)originalSolution.getPosition(i).x() * scaleFactor), qRound((qreal)originalSolution.getPosition(i).y() * scaleFactor));
		newSolution.setPosition(i, finePos);
	}
}

void RasterStripPackingSolverDoubleGLS::updateMapsLength(int pixelWidth, RasterStripPackingParameters &params) {
	int deltaPixels = this->currentWidth - pixelWidth;
	for (int itemId = 0; itemId < originalProblem->count(); itemId++)
	for (uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		//curMap->shrink(qRound(deltaPixels*this->searchProblem->getScale()/this->originalProblem->getScale()));

		//int newTotalOverlapMapWidth = curIfp->width() - (this->initialWidth - pixelWidth);
		//newTotalOverlapMapWidth = qCeil(newTotalOverlapMapWidth*this->searchProblem->getScale() / this->originalProblem->getScale());

		std::shared_ptr<RasterNoFitPolygon> curIfp = this->originalProblem->getIfps()->getRasterNoFitPolygon(-1, -1, this->originalProblem->getItemType(itemId), angle);
		int leftMostPointX = curIfp->width() - curIfp->getOriginX() - 1 - (this->initialWidth - pixelWidth);
		int newLeftMostPointX = qCeil(leftMostPointX*this->searchProblem->getScale() / this->originalProblem->getScale());
		int newTotalOverlapMapWidth = newLeftMostPointX + curMap->getReferencePoint().x() + 1;
		curMap->setWidth(newTotalOverlapMapWidth);

		size_t curMapMem = curMap->getWidth()*curMap->getHeight()*sizeof(qreal);
	}

	currentWidth = pixelWidth;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolverDoubleGLS::getTotalOverlapSearchMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < searchProblem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(searchProblem->getNfps()->getRasterNoFitPolygon(searchProblem->getItemType(i), solution.getOrientation(i), searchProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	return currrentPieceMap;
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolverDoubleGLS::getMinimumOverlapSeachPosition(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	// Scale solution to seach scale
	RasterPackingSolution roughSolution;
	qreal zoomFactor = this->originalProblem->getScale() / this->searchProblem->getScale();
	getScaledSolution(solution, roughSolution, 1.0 / zoomFactor);

	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapSearchMap(itemId, orientation, roughSolution, params);
	//float fvalue = value;
	float fvalue;
	QPoint minRelativePos = map->getMinimum(fvalue, params.getPlacementCriteria());

	// Rescale position before returning
	return zoomFactor*(minRelativePos - map->getReferencePoint());
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolverDoubleGLS::getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution) {
	// Determine zoomed area inside the innerfit polygon
	std::shared_ptr<RasterNoFitPolygon> curIfp = this->originalProblem->getIfps()->getRasterNoFitPolygon(-1, -1, this->originalProblem->getItemType(itemId), orientation);
	QRect curIfpBoundingBox( QPoint(-curIfp->getOriginX(), -curIfp->getOriginY()), QSize(curIfp->width() - (this->initialWidth - this->currentWidth), curIfp->height()));
	QRect zoomSquareRect(QPoint(pos.x() - width / 2, pos.y() - height / 2), QSize(width, height));
	zoomSquareRect = zoomSquareRect.intersected(curIfpBoundingBox);

	// Create zoomed overlap Map
	std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(zoomSquareRect));
	for (int j = zoomSquareRect.top(); j <= zoomSquareRect.bottom(); j++)
		for (int i = zoomSquareRect.left(); i <= zoomSquareRect.right(); i++)
			curZoomedMap->setValue(QPoint(i, j), getTotalOverlapMapSingleValue(itemId, orientation, QPoint(i, j), solution, this->originalProblem));

	return curZoomedMap;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolverDoubleGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QPoint pos = getMinimumOverlapSeachPosition(itemId, orientation, solution, params);
	int zoomSquareSize = ZOOMNEIGHBORHOOD*qRound(this->originalProblem->getScale() / this->searchProblem->getScale());
	return getRectTotalOverlapMap(itemId, orientation, pos, zoomSquareSize, zoomSquareSize, solution);
}