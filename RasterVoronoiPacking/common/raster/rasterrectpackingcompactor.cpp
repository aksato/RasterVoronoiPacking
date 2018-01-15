#include "rasterrectpackingcompactor.h"

using namespace RASTERVORONOIPACKING;

QPair<int, int> RasterRectangularPackingCompactor::setContainerDimensions(int newLength, int newHeight) {
	// Check if size is smaller than smallest item width
	if (newLength < this->problem->getMaxWidth()) { newLength = this->problem->getMaxWidth(); }
	if (newHeight < this->problem->getMaxHeight()) { newHeight = this->problem->getMaxHeight(); }

	// Resize container
	overlapEvaluator->updateMapsDimensions(newLength, newHeight);

	return QPair<int, int>(newLength, newHeight);
}

bool RasterRectangularPackingCompactor::getShrinkedDimension(qreal realDim, qreal &newRealDim, int minimumDimension) {
	// Check if dimension is already minimum
	if (qRound(realDim) == minimumDimension) return false;

	// Get next dimension and check limit
	if (qRound(newRealDim) < minimumDimension) newRealDim = minimumDimension;

	return true;
}

void RasterRectangularPackingRandomCompactor::randomShrinkDimensions(bool changeLength, qreal ratio) {
	if (changeLength) {
		qreal reducedRealLength = ratio * curRealLength;
		if (getShrinkedDimension(curRealLength, reducedRealLength, this->problem->getMaxWidth())) {
			curRealLength = reducedRealLength;
			return;
		}
	}
	qreal reducedRealHeight = ratio * curRealHeight;
	if (getShrinkedDimension(curRealHeight, reducedRealHeight, this->problem->getMaxHeight())) {
		curRealHeight = reducedRealHeight;
		return;
	}
	qreal reducedRealLength = ratio * curRealLength;
	if (getShrinkedDimension(curRealLength, reducedRealLength, this->problem->getMaxWidth())) {
		curRealLength = reducedRealLength;
	}
}

bool RasterRectangularPackingRandomCompactor::shrinkContainer(RasterPackingSolution &solution) {
	// Update best area
	bool bestSol = false;
	qreal curArea = curRealLength * curRealHeight;
	if (curArea < bestArea) {
		bestArea = curArea;
		bestSol = true;
	}

	// Reduction
	randomShrinkDimensions(qrand() % 2 - 1, 1 - rdec);

	// Set new dimensions
	QPair<int, int> newDimensions = setContainerDimensions(qRound(curRealLength), qRound(curRealHeight));
	curRealLength = newDimensions.first; curRealHeight = newDimensions.second;

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealLength)) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->problem->getContainerHeight() - qRound(curRealHeight)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}

	return bestSol;
}

bool RasterRectangularPackingRandomCompactor::expandContainer(RasterPackingSolution &solution) {
	qreal ratio = 1 + rdec;
	bool changeLength = qrand() % 2 - 1;
	// Expansion
	if (changeLength) { curRealLength = ratio * curRealLength; }
	else { curRealHeight = ratio * curRealHeight; }
	setContainerDimensions(qRound(curRealLength), qRound(curRealHeight));

	return true;
}

void RasterRectangularPackingCompactor::generateRandomSolution(RasterPackingSolution &solution) {
	for (int i = 0; i < problem->count(); i++)  {
		// Shuffle angle
		int totalAngles = problem->getItem(i)->getAngleCount();
		int rnd_angle = 0;
		if (totalAngles != 0) {
			rnd_angle = qrand() % ((totalAngles - 1 + 1) - 0) + 0;
			solution.setOrientation(i, rnd_angle);
		}

		// Shuffle position
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(i), rnd_angle);
		int newIfpWidth = ifp->width() - qRound(problem->getScale() * (qreal)(this->problem->getContainerWidth() - qRound(curRealLength)) / this->problem->getScale());
		int newIfpHeight = ifp->height() - qRound(problem->getScale() * (qreal)(this->problem->getContainerHeight() - qRound(curRealHeight)) / this->problem->getScale());
		int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
		int maxX = minX + newIfpWidth - 1;
		int maxY = minY + newIfpHeight - 1;

		int rnd_x = qrand() % ((maxX + 1) - minX) + minX;
		int rnd_y = qrand() % ((maxY + 1) - minY) + minY;
		solution.setPosition(i, QPoint(rnd_x, rnd_y));
	}
}

void RasterRectangularPackingCompactor::generateBottomLeftSolution(RasterPackingSolution &solution) {
	// Create bottom left layout
	int layoutLength, layoutHeight;
	generateBottomLeftLayout(solution, layoutLength, layoutHeight);

	// Set new dimension for the square container
	QPair<int, int> newDimensions = setContainerDimensions(layoutLength, layoutHeight);
	curRealLength = newDimensions.first; curRealHeight = newDimensions.second;
}

bool RasterRectangularPackingBagpipeCompactor::expandContainer(RasterPackingSolution &solution) {
	// Keep aspect ratio
	qreal ratio = 1 + rinc;
	qreal expandedRealLength = sqrt(ratio) * curRealLength;
	qreal expandedRealHeight = sqrt(ratio) * curRealHeight;
	if (qRound(expandedRealLength) * qRound(expandedRealHeight) < bestArea) {
		curRealHeight = expandedRealHeight; curRealLength = expandedRealLength;
		// Set new dimensions
		QPair<int, int> newDimensions = setContainerDimensions(qRound(curRealLength), qRound(curRealHeight));
	}
	else {
		qreal rinc = ratio - 1.0;
		qreal expansionDelta = qMax(rinc*curRealLength, rinc*curRealHeight);
		qreal curArea = curRealLength * curRealHeight;
		if (bagpipeDirection) {
			expandedRealHeight = curRealHeight + expansionDelta;
			qreal reducedRealLength = curArea / expandedRealHeight;
			getShrinkedDimension(curRealLength, reducedRealLength, this->problem->getMaxWidth());

			curRealLength = reducedRealLength;
			curRealHeight = expandedRealHeight;

			if (qRound(curRealLength) <= this->problem->getMaxWidth()) { bagpipeDirection = !bagpipeDirection; qDebug() << "YOU'VE GOT BAGPIPED!"; }
		}
		else {
			expandedRealLength = curRealLength + expansionDelta;
			qreal reducedRealHeight = curArea / expandedRealLength;
			getShrinkedDimension(curRealHeight, reducedRealHeight, this->problem->getMaxHeight());

			curRealLength = expandedRealLength;
			curRealHeight = reducedRealHeight;
			if (qRound(curRealHeight) <= this->problem->getMaxHeight()){ bagpipeDirection = !bagpipeDirection; qDebug() << "YOU'VE GOT BAGPIPED!"; }
		}
		// Set new dimensions
		QPair<int, int> newDimensions = setContainerDimensions(qRound(curRealLength), qRound(curRealHeight));
		//curRealLength = newDimensions.first; curRealHeight = newDimensions.second;
	}


	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealLength)) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->problem->getContainerHeight() - qRound(curRealHeight)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}
	return true;
}


bool RasterRectangularPackingBagpipeCompactor::shrinkContainer(RasterPackingSolution &solution) {
	// Update best area
	bool bestSol = false;
	qreal curArea = curRealLength * curRealHeight;
	if (curArea < bestArea) {
		bestArea = curArea;
		bestSol = true;
	}

	// Reduction
	qreal ratio = sqrt(1 - rdec);
	qreal reducedRealLength = ratio * curRealLength;
	qreal reducedRealHeight = ratio * curRealHeight;
	if (getShrinkedDimension(curRealLength, reducedRealLength, this->problem->getMaxWidth())) curRealLength = reducedRealLength;
	if (getShrinkedDimension(curRealLength, reducedRealLength, this->problem->getMaxWidth())) curRealHeight = reducedRealHeight;

	// Set new dimensions
	QPair<int, int> newDimensions = setContainerDimensions(qRound(curRealLength), qRound(curRealHeight));
	curRealLength = newDimensions.first; curRealHeight = newDimensions.second;

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealLength)) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->problem->getContainerHeight() - qRound(curRealHeight)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}

	// Update best area
	return bestSol;
}

void RasterRectangularPackingCompactor::setContainerDimensions(int newLength, int newHeight, RasterPackingSolution &solution) {
	QPair<int, int> newDimensions = setContainerDimensions(newLength, newHeight);
	curRealLength = newDimensions.first; curRealHeight = newDimensions.second;

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealLength)) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->problem->getContainerHeight() - qRound(curRealHeight)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}
}