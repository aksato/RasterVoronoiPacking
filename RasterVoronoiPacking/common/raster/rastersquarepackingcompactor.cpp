#include "rastersquarepackingcompactor.h"

using namespace RASTERVORONOIPACKING;

int RasterSquarePackingCompactor::setContainerSize(int newSize) {
	// Check if size is smaller than smallest item width
	if (newSize < this->problem->getMaxWidth()) { newSize = this->problem->getMaxWidth(); }
	if (newSize < this->problem->getMaxHeight()) { newSize = this->problem->getMaxHeight(); }

	// Resize container
	overlapEvaluator->updateMapsDimensions(newSize, newSize);
	return newSize;
}

bool RasterSquarePackingCompactor::shrinkContainer(RasterPackingSolution &solution) {
	// Minimum length obtained
	if (this->problem->getMaxWidth() == bestSize || this->problem->getMaxHeight() == bestSize) return false;

	// Update best size
	bestSize = qRound(curRealSize);

	// Set new size
	int newSize = qRound(sqrt(1.0 - rdec)*curRealSize);
	if (newSize == qRound(curRealSize)) newSize--;
	curRealSize = setContainerSize(newSize);

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < problem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(0, 0, problem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->problem->getContainerWidth() - qRound(curRealSize)) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->problem->getContainerHeight() - qRound(curRealSize)) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}

	return true;
}


bool RasterSquarePackingCompactor::expandContainer(RasterPackingSolution &solution) {
	int previousSize = qRound(curRealSize);
	// Determine new size
	if (qRound(sqrt(1.0 + rinc)*curRealSize) >= bestSize) {
		curRealSize = (curRealSize + (qreal)bestSize) / 2.0;
		if (qRound(curRealSize) == bestSize) curRealSize = bestSize - 1;
	}
	else curRealSize = sqrt(1.0 + rinc)*curRealSize;

	// Set new width
	if (previousSize == qRound(curRealSize)) return false;
	setContainerSize(qRound(curRealSize));
	return true;
}

void RasterSquarePackingCompactor::generateRandomSolution(RasterPackingSolution &solution) {
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
		int newIfpWidth = ifp->width() - qRound(problem->getScale() * (qreal)(this->problem->getContainerWidth() - qRound(curRealSize)) / this->problem->getScale());
		int newIfpHeight = ifp->height() - qRound(problem->getScale() * (qreal)(this->problem->getContainerHeight() - qRound(curRealSize)) / this->problem->getScale());
		int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
		int maxX = minX + newIfpWidth - 1;
		int maxY = minY + newIfpHeight - 1;

		int rnd_x = qrand() % ((maxX + 1) - minX) + minX;
		int rnd_y = qrand() % ((maxY + 1) - minY) + minY;
		solution.setPosition(i, QPoint(rnd_x, rnd_y));
	}
}

void RasterSquarePackingCompactor::generateBottomLeftLayout(RasterPackingSolution &solution, int &length, int &height) {
	QVector<int> sequence;
	for (int i = 0; i < problem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	int layoutLength = 0;
	int layoutHeight = 0;
	for (int k = 0; k < problem->count(); k++) {
		int shuffledId = sequence[k];
		int minItemArea, curBestLayoutLength, curBestLayoutHeight;
		// Find left bottom placement for item
		for (unsigned int angle = 0; angle < problem->getItem(shuffledId)->getAngleCount(); angle++) {
			//qDebug() << "Item" << k << "Angle" << angle;
			// Get IFP bounding box
			int  minIfpX, minIfpY, maxIfpX, maxIfpY;
			problem->getIfpBoundingBox(shuffledId, angle, minIfpX, minIfpY, maxIfpX, maxIfpY);
			QPoint curPos(minIfpX, minIfpY);
			int i = 0;
			while (1) {
				int aux = 0;
				bool exit = false;
				for (int j = 0; j < i; j++){
					curPos = QPoint(minIfpX + i, minIfpY + aux);
					//qDebug() << curPos;
					if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution)) { exit = true;  break; }
					curPos = QPoint(minIfpX + aux, minIfpY + i);
					//qDebug() << curPos;
					if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution)) { exit = true;  break; }
					aux++;
				}
				if (exit) break;
				curPos = QPoint(minIfpX + i, minIfpY + i);
				//qDebug() << curPos;
				if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution)) break;
				i++;
			}
			// Check minimum X and Y coordinate
			int maxItemY = curPos.y() + qRound(problem->getScale()*problem->getItem(shuffledId)->getMaxY(angle));
			int maxItemX = curPos.x() + qRound(problem->getScale()*problem->getItem(shuffledId)->getMaxX(angle));
			int curArea = qMax(maxItemX, layoutLength) * qMax(maxItemY, layoutHeight);
			if (angle == 0 || curArea < minItemArea) {
				minItemArea = curArea;
				solution.setPosition(shuffledId, curPos); solution.setOrientation(shuffledId, angle);
				curBestLayoutLength = qMax(maxItemX, layoutLength);
				curBestLayoutHeight = qMax(maxItemY, layoutHeight);
			}
		}
		layoutLength = curBestLayoutLength;
		layoutHeight = curBestLayoutHeight;
	}
	length = layoutLength; height = layoutHeight;
}

void RasterSquarePackingCompactor::generateBottomLeftSolution(RasterPackingSolution &solution) {
	// Create bottom left layout
	int layoutLength, layoutHeight;
	generateBottomLeftLayout(solution, layoutLength, layoutHeight);

	// Set new dimension for the square container
	int largestDim = layoutLength > layoutHeight ? layoutLength : layoutHeight;
	bestSize = setContainerSize(largestDim);
	curRealSize = bestSize;
}

bool RasterSquarePackingCompactor::detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution) {
	if (itemSequencePos == 0) return false;
	int itemId = sequence[itemSequencePos];
	for (int i = 0; i < itemSequencePos; i++) {
		int curFixedItemId = sequence[i];
		if (problem->areOverlapping(itemId, itemPos, itemAngle, curFixedItemId, solution.getPosition(curFixedItemId), solution.getOrientation(curFixedItemId)))
			return true;
	}
	return false;
}