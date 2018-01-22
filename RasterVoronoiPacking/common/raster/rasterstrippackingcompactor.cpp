#include "rasterstrippackingcompactor.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

int RasterStripPackingCompactor::setContainerWidth(int newWitdh) {
	// Check if size is smaller than smallest item width
	if (newWitdh < this->problem->getMaxWidth()) newWitdh = this->problem->getMaxWidth();

	// Resize container
	overlapEvaluator->updateMapsLength(newWitdh);
	return newWitdh;
}

bool RasterStripPackingCompactor::shrinkContainer(RasterPackingSolution &solution) {
	// Minimum length obtained
	if (this->problem->getMaxWidth() == bestWidth) return false;

	//Update best length
	bestWidth = qRound(curRealLength);

	// Set new width
	int newLength = qRound((1.0 - rdec)*curRealLength);
	if (newLength == qRound(curRealLength)) newLength--;
	curRealLength = setContainerWidth(newLength);

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = this->problem->getIfps()->getRasterNoFitPolygon(0, 0, this->problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - curRealLength) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) {
			curItemPos.setX(maxPositionX);
			solution.setPosition(itemId, curItemPos);
		}
	}

	return true;
}

bool RasterStripPackingCompactor::expandContainer(RasterPackingSolution &solution) {
	int previousLength = qRound(curRealLength);
	// Determine new width
	if (qRound((1.0 + rinc)*curRealLength) >= bestWidth) {
		curRealLength = (curRealLength + (qreal)bestWidth) / 2.0;
		if (qRound(curRealLength) == bestWidth) curRealLength = bestWidth - 1;
	}
	else curRealLength = (1.0 + rinc)*curRealLength;

	// Set new width
	if (previousLength == qRound(curRealLength)) return false;
	setContainerWidth(qRound(curRealLength));
	return true;
}

qreal RasterStripPackingCompactor::getItemMaxDimension(int itemId) {
	qreal maxDim = 0;
	for (unsigned int angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
		qreal curMaxDim = problem->getItem(itemId)->getMaxX(angle) - problem->getItem(itemId)->getMinX(angle);
		if (curMaxDim > maxDim) maxDim = curMaxDim;
	}
	return maxDim;
}

void RasterStripPackingCompactor::generateRandomSolution(RasterPackingSolution &solution) {
	for(int i =0; i < problem->count(); i++)  {
	    // Shuffle angle
	    int totalAngles = problem->getItem(i)->getAngleCount();
	    int rnd_angle = 0;
	    if(totalAngles != 0) {
	        rnd_angle = qrand() % ((totalAngles -1 + 1) - 0) + 0;
	        solution.setOrientation(i, rnd_angle);
	    }
	
	    // Shuffle position
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(i), rnd_angle);
		int newIfpWidth = ifp->width() - qRound(problem->getScale() * (qreal)(this->problem->getContainerWidth() - qRound(curRealLength)) / this->problem->getScale());
	    int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
	    int maxX = minX + newIfpWidth - 1;
	    int maxY = minY + ifp->height() - 1;
	
	    int rnd_x =  qrand() % ((maxX + 1) - minX) + minX;
	    int rnd_y =  qrand() % ((maxY + 1) - minY) + minY;
	    solution.setPosition(i, QPoint(rnd_x, rnd_y));
	}
}

void RasterStripPackingCompactor::generateBottomLeftSolution(RasterPackingSolution &solution) {
	QVector<int> sequence;
	for (int i = 0; i < this->problem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	int layoutLength = 0;
	QList<int> placedItems;
	int partialLength = 0;
	for (int k = 0; k < this->problem->count(); k++) {
		int shuffledId = sequence[k];
		int minMaxItemX;

		// Resize container to maximum
		int angle = 0;
		int curMaxItemLength = qCeil(this->problem->getScale()*getItemMaxDimension(shuffledId));
		int partialLayoutMaximumLength = curMaxItemLength + layoutLength;
		int newWidth = partialLayoutMaximumLength < this->problem->getMaxWidth() ? this->problem->getMaxWidth() : partialLayoutMaximumLength;
		this->setContainerWidth(newWidth);

		// Find best orientation
		for (unsigned int angle = 0; angle < this->problem->getItem(shuffledId)->getAngleCount(); angle++) {
			QPoint curPos = overlapEvaluator->getBottomLeftPosition(shuffledId, angle, solution, placedItems);
			// Check minimum X coordinate
			int maxItemX = curPos.x() + qCeil(this->problem->getScale()*this->problem->getItem(shuffledId)->getMaxX(angle));
			if (angle == 0 || maxItemX < minMaxItemX) {
				minMaxItemX = maxItemX;
				solution.setPosition(shuffledId, curPos); solution.setOrientation(shuffledId, angle);
			}
		}
		if (minMaxItemX > layoutLength) layoutLength = minMaxItemX;
		placedItems << shuffledId;
	}

	// Set new dimension for the container
	setContainerWidth(layoutLength);
	curRealLength = layoutLength;
	bestWidth = layoutLength;
}

// DEBUG ONLY
void RasterStripPackingCompactor::setContainerWidth(int newWitdh, RasterPackingSolution &solution) {
	// Set new width
	curRealLength = setContainerWidth(newWitdh);

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = this->problem->getIfps()->getRasterNoFitPolygon(0, 0, this->problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealLength)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) {
			curItemPos.setX(maxPositionX);
			solution.setPosition(itemId, curItemPos);
		}
	}
}