#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

#define ZOOMNEIGHBORHOOD 3

void RasterStripPackingSolver::setProblem(std::shared_ptr<RasterPackingProblem> _problem) {
    this->originalProblem = _problem;
    //this->currentProblem = this->originalProblem;

    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
        for(uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
            std::shared_ptr<TotalOverlapMap> curMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(originalProblem->getIfps()->getRasterNoFitPolygon(-1,-1,originalProblem->getItemType(itemId),angle)));			
            maps.addOverlapMap(itemId,angle,curMap);
            // FIXME: Delete innerift polygons as they are used to release memomry
        }
    }
	currentWidth = this->originalProblem->getContainerWidth(); currentHeight = this->originalProblem->getContainerHeight();
	initialWidth = currentWidth; initialHeight = currentHeight;
}

// Basic Functions
void RasterStripPackingSolver::generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
    for(int i =0; i < originalProblem->count(); i++)  {
        // Shuffle angle
        int totalAngles = originalProblem->getItem(i)->getAngleCount();
        int rnd_angle = 0;
        if(totalAngles != 0) {
            rnd_angle = qrand() % ((totalAngles -1 + 1) - 0) + 0;
            solution.setOrientation(i, rnd_angle);
        }

        // Shuffle position
		std::shared_ptr<RasterNoFitPolygon> ifp = originalProblem->getIfps()->getRasterNoFitPolygon(-1, -1, originalProblem->getItemType(i), rnd_angle);
		int newIfpWidth = ifp->width() - qRound(originalProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
        int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
        int maxX = minX + newIfpWidth - 1;
        int maxY = minY + ifp->height() - 1;

        int rnd_x =  qrand() % ((maxX + 1) - minX) + minX;
        int rnd_y =  qrand() % ((maxY + 1) - minY) + minY;
        solution.setPosition(i, QPoint(rnd_x, rnd_y));
    }
}

// --> Generate initial solution using the bottom left heuristic and resize the container accordingly
void RasterStripPackingSolver::generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params, BottomLeftMode mode) {
	switch(mode) {
		case BL_STRIPPACKING: generateBottomLeftStripSolution(solution, params); break;
		case BL_RECTANGULAR: generateBottomLeftRectangleSolution(solution, params); break;
		case BL_SQUARE: generateBottomLeftSquareSolution(solution, params); break;
	}
}

void RasterStripPackingSolver::generateBottomLeftStripSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	int layoutLength = 0;
	for (int k = 0; k < originalProblem->count(); k++) {
		int shuffledId = sequence[k];
		int minMaxItemX;
		// Find left bottom placement for item
		for (unsigned int angle = 0; angle < originalProblem->getItem(shuffledId)->getAngleCount(); angle++) {
			// Get IFP bounding box
			int  minIfpX, minIfpY, maxIfpX, maxIfpY;
			getIfpBoundingBox(shuffledId, angle, minIfpX, minIfpY, maxIfpX, maxIfpY, originalProblem);
			QPoint curPos(minIfpX, minIfpY);
			while (1) { // FIXME: Infinite loop?
				//if (qFuzzyCompare(1.0 + 0.0, 1.0 + overlap)) break;
				if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution, originalProblem))
					break;
				// Get next position
				curPos.setY(curPos.y() + 1); if (curPos.y() > maxIfpY) { curPos.setY(minIfpY); curPos.setX(curPos.x() + 1); }
			}
			// Check minimum X coordinate
			int maxItemX = getItemMaxX(curPos.x(), angle, shuffledId, originalProblem);
			if (angle == 0 || maxItemX < minMaxItemX) {
				minMaxItemX = maxItemX;
				solution.setPosition(shuffledId, curPos); solution.setOrientation(shuffledId, angle);
			}
		}
		if (minMaxItemX > layoutLength) layoutLength = minMaxItemX;
	}
	setContainerWidth(layoutLength, solution, params);
}

void RASTERVORONOIPACKING::getIfpBoundingBox(int itemId, int angle, int &bottomLeftX, int &bottomLeftY, int &topRightX, int &topRightY, std::shared_ptr<RasterPackingProblem> problem) {
	std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(-1, -1, problem->getItemType(itemId), angle);
	bottomLeftX = -ifp->getOriginX();
	bottomLeftY = -ifp->getOriginY();
	topRightX = bottomLeftX + ifp->width() - 1;
	topRightY = bottomLeftY + ifp->height() - 1;
}

// Detect if item is in overlapping position for a subset of fixed items
bool RasterStripPackingSolver::detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	if (itemSequencePos == 0) return false;
	int itemId = sequence[itemSequencePos];
	for (int i = 0; i < itemSequencePos; i++) {
		int curFixedItemId = sequence[i];
		if (detectOverlap(itemId, itemPos, itemAngle, curFixedItemId, solution.getPosition(curFixedItemId), solution.getOrientation(curFixedItemId), problem))
			return true;
	}
	return false;
}

// FIXME: use discretization from nfp/ifp
int RasterStripPackingSolver::getItemMaxX(int posX, int angle, int itemId, std::shared_ptr<RasterPackingProblem> problem) {
	int itemMinX, itemMaxX, itemMinY, itemMaxY; problem->getItem(itemId)->getBoundingBox(itemMinX, itemMaxX, itemMinY, itemMaxY);
	int realItemMaxX;
	if (problem->getItem(itemId)->getAngleValue(angle) == 0) realItemMaxX = itemMaxX;
	if (problem->getItem(itemId)->getAngleValue(angle) == 90) realItemMaxX = -itemMinY;
	if (problem->getItem(itemId)->getAngleValue(angle) == 180) realItemMaxX = -itemMinX;
	if (problem->getItem(itemId)->getAngleValue(angle) == 270) realItemMaxX = itemMaxY;
	return posX + qRound((qreal)realItemMaxX*problem->getScale());

}

// --> Get layout overlap (sum of individual overlap values)
qreal RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
    qreal totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		totalOverlap += getItemTotalOverlap(itemId, solution, originalProblem);
    }
    return totalOverlap;
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value, PositionChoice placementHeuristic) {
	float fvalue = value;
	QPoint minRelativePos = map->getMinimum(fvalue, placementHeuristic);
    return minRelativePos - map->getReferencePoint();
}

// --> Change container size
void RasterStripPackingSolver::updateMapsLength(int pixelWidth, RasterStripPackingParameters &params) {
    int deltaPixels = this->currentWidth - pixelWidth;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++)
        for(uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
            std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
            curMap->shrink(deltaPixels);
			size_t curMapMem = curMap->getWidth()*curMap->getHeight()*sizeof(qreal);
        }

    currentWidth = pixelWidth;
}

bool RasterStripPackingSolver::setContainerWidth(int &pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	// Check if size is smaller than smallest item width
	if (this->getMinimumContainerWidth() <= this->initialWidth - pixelWidth) { pixelWidth = this->currentWidth; return false; }

	// Resize container
	updateMapsLength(pixelWidth, params);

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = originalProblem->getIfps()->getRasterNoFitPolygon(-1, -1, originalProblem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->initialWidth - this->currentWidth) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) {
			curItemPos.setX(maxPositionX);
			solution.setPosition(itemId, curItemPos);
		}
	}

	return true;
}

// --> Get nfp distance value: pos1 is static item and pos2 is orbiting item
qreal getNfpValue(QPoint pos1, QPoint pos2, std::shared_ptr<RasterNoFitPolygon> curNfp, bool &isZero) {
    isZero = false;
    QPoint relPos = pos2 - pos1 + curNfp->getOrigin();

//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) {
    if(relPos.x() < 0 || relPos.x() > curNfp->width()-1 || relPos.y() < 0 || relPos.y() > curNfp->height()-1) {
        isZero = true;
        return 0.0;
    }

//    int indexValue = curNfp->getImage().pixelIndex(relPos);
    int indexValue = curNfp->getPixel(relPos.x(), relPos.y());
    if(indexValue == 0) {
        isZero = true;
        return 0.0;
    }
    return 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;
}

// --> Get two items minimum overlap
qreal RasterStripPackingSolver::getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem) {
    std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
    qreal value1Static2Orbiting, value2Static1Orbiting;
    bool feasible;

	curNfp1Static2Orbiting = problem->getNfps()->getRasterNoFitPolygon(
        originalProblem->getItemType(itemId1), orientation1,
        originalProblem->getItemType(itemId2), orientation2);
    value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
    if(feasible) return 0.0;

	curNfp2Static1Orbiting = problem->getNfps()->getRasterNoFitPolygon(
            originalProblem->getItemType(itemId2), orientation2,
            originalProblem->getItemType(itemId1), orientation1);
    value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
    if(feasible) return 0.0;

    return value1Static2Orbiting < value2Static1Orbiting ? value1Static2Orbiting : value2Static1Orbiting;
}

bool RasterStripPackingSolver::detectOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2, std::shared_ptr<RasterPackingProblem> problem) {
	std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
	qreal value1Static2Orbiting, value2Static1Orbiting;
	bool feasible;

	curNfp1Static2Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		originalProblem->getItemType(itemId1), orientation1,
		originalProblem->getItemType(itemId2), orientation2);
	value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
	if (feasible) return false;

	curNfp2Static1Orbiting = problem->getNfps()->getRasterNoFitPolygon(
		originalProblem->getItemType(itemId2), orientation2,
		originalProblem->getItemType(itemId1), orientation1);
	value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
	if (feasible) return false;

	if (qFuzzyCompare(1.0 + value1Static2Orbiting, 1.0) || qFuzzyCompare(1.0 + value2Static1Orbiting, 1.0))
		return false;
	return true;
}

qreal RasterStripPackingSolver::getItemTotalOverlap(int itemId, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
    qreal totalOverlap = 0;
    for(int i =0; i < originalProblem->count(); i++) {
        if(i == itemId) continue;
        totalOverlap += getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId),
			i, solution.getPosition(i), solution.getOrientation(i), problem);
    }
    return totalOverlap;
}

qreal RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps, RasterStripPackingParameters &params) {
    qreal totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		qreal itemOverlap = getItemTotalOverlap(itemId, solution, originalProblem);
        individualOverlaps.append(itemOverlap);
        totalOverlap += itemOverlap;
    }
    return totalOverlap;
}

void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	for (int i = 0; i < originalProblem->count(); i++) {
		int shuffledId = sequence[i];
		if (qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution, this->originalProblem))) continue;
		qreal minValue; QPoint minPos; int minAngle = 0;
		minPos = getMinimumOverlapPosition(shuffledId, minAngle, solution, minValue, params);
		for (uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
			qreal curValue; QPoint curPos;
			curPos = getMinimumOverlapPosition(shuffledId, curAngle, solution, curValue, params);
			if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
		}
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapMap(itemId, orientation, solution, params);
	//float fvalue = value;
	float fvalue;
	QPoint minRelativePos = map->getMinimum(fvalue, params.getPlacementCriteria());
	value = fvalue;
	return minRelativePos - map->getReferencePoint();
}

qreal RasterStripPackingSolver::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	qreal totalOverlap = 0;
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i), problem);
	}
	return totalOverlap;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		currrentPieceMap->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i));
	}
	return currrentPieceMap;
}

void getNextBLPosition(QPoint &curPos, int  minIfpX, int minIfpY, int maxIfpX, int maxIfpY) {
	curPos.setY(curPos.y() + 1); 
	if (curPos.y() > maxIfpY) { 
		curPos.setY(minIfpY); curPos.setX(curPos.x() + 1);
	}
}