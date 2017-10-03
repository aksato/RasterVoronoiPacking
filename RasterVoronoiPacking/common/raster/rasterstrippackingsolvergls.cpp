#include "rasterstrippackingsolvergls.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;


void RasterStripPackingSolverGLS::updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<WeightIncrement> solutionOverlapValues;
	qreal maxOValue = 0;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < originalProblem->count(); itemId1++)
	for (int itemId2 = 0; itemId2 < originalProblem->count(); itemId2++) {
		if (itemId1 == itemId2) continue;
		qreal curOValue = getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
			itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2), originalProblem);
		if (curOValue != 0) {
			solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, (qreal)curOValue));
			if (curOValue > maxOValue) maxOValue = curOValue;
		}
	}

	// Divide vector by maximum
	std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value / maxOValue; });

	// Add to the current weight map
	glsWeights->updateWeights(solutionOverlapValues);
}

//  TODO: Update cache information!
void RasterStripPackingSolverGLS::resetWeights() {
	glsWeights->reset(originalProblem->count());
}

qreal RasterStripPackingSolverGLS::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	qreal totalOverlap = 0;
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += glsWeights->getWeight(itemId, i)*getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i), problem);
	}
	return totalOverlap;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolverGLS::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
	return currrentPieceMap;
}


// FIXME: use discretization from nfp/ifp
int RasterStripPackingSolver2D::getItemMaxY(int posY, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem) {
	int itemMinX, itemMaxX, itemMinY, itemMaxY; problem->getItem(itemId)->getBoundingBox(itemMinX, itemMaxX, itemMinY, itemMaxY);
	int realItemMaxY;
	if (problem->getItem(itemId)->getAngleValue(angle) == 0) realItemMaxY = itemMaxY;
	if (problem->getItem(itemId)->getAngleValue(angle) == 90) realItemMaxY = itemMaxX;
	if (problem->getItem(itemId)->getAngleValue(angle) == 180) realItemMaxY = -itemMinY;
	if (problem->getItem(itemId)->getAngleValue(angle) == 270) realItemMaxY = -itemMinX;
	return posY + qRound((qreal)realItemMaxY*problem->getScale());

}

// --> Generate initial solution using the bottom left heuristic and resize the container accordingly
void RasterStripPackingSolver2D::generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	int layoutLength = 0;
	int layoutHeight = 0;
	for (int k = 0; k < originalProblem->count(); k++) {
		int shuffledId = sequence[k];
		int minItemArea, curBestLayoutLength, curBestLayoutHeight;
		// Find left bottom placement for item
		for (unsigned int angle = 0; angle < originalProblem->getItem(shuffledId)->getAngleCount(); angle++) {
			//qDebug() << "Item" << k << "Angle" << angle;
			// Get IFP bounding box
			int  minIfpX, minIfpY, maxIfpX, maxIfpY;
			getIfpBoundingBox(shuffledId, angle, minIfpX, minIfpY, maxIfpX, maxIfpY, originalProblem);
			QPoint curPos(minIfpX, minIfpY);
			int i = 0;
			while (1) {
				int aux = 0;
				bool exit = false;
				for (int j = 0; j < i; j++){
					curPos = QPoint(minIfpX + i, minIfpY + aux);
					//qDebug() << curPos;
					if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution, originalProblem)) { exit = true;  break; }
					curPos = QPoint(minIfpX + aux, minIfpY + i);
					//qDebug() << curPos;
					if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution, originalProblem)) { exit = true;  break; }
					aux++;
				}
				if (exit) break;
				curPos = QPoint(minIfpX + i, minIfpY + i);
				//qDebug() << curPos;
				if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution, originalProblem)) break;
				i++;
			}
			// Check minimum X and Y coordinate
			int maxItemY = getItemMaxY(curPos.y(), angle, shuffledId, originalProblem);
			int maxItemX = getItemMaxX(curPos.x(), angle, shuffledId, originalProblem);
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
	//setContainerWidth(layoutLength, solution, params);
	setContainerDimensions(layoutLength, layoutHeight, solution, params);
}

void RasterStripPackingSolver2D::generateBottomLeftSquareSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	generateBottomLeftSolution(solution, params);
	int largestDim = currentWidth > currentHeight ? currentWidth : currentHeight;
	setContainerDimensions(largestDim, largestDim, solution, params);
}

bool RasterStripPackingSolver2D::setContainerDimensions(int &pixelWidthX, int &pixelWidthY, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	// Resize container
	updateMapsDimensions(pixelWidthX, pixelWidthY, params);

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = originalProblem->getIfps()->getRasterNoFitPolygon(-1, -1, originalProblem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->initialWidth - this->currentWidth) - 1;
		int maxPositionY = -ifp->getOriginY() + ifp->height() - (this->initialHeight - this->currentHeight) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) curItemPos.setX(maxPositionX);
		if (curItemPos.y() > maxPositionY) curItemPos.setY(maxPositionY);
		solution.setPosition(itemId, curItemPos);
	}

	return true;
}

// --> Change container size
void RasterStripPackingSolver2D::updateMapsDimensions(int pixelWidth, int pixelHeight, RasterStripPackingParameters &params) {
	int deltaPixelsX = this->currentWidth - pixelWidth;
	int deltaPixelsY = this->currentHeight - pixelHeight;
	for (int itemId = 0; itemId < originalProblem->count(); itemId++)
	for (uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
		curMap->shrink2D(deltaPixelsX, deltaPixelsY);
	}
	currentWidth = pixelWidth;
	currentHeight = pixelHeight;
}