#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>
#include "../cuda/gpuinfo.h"

using namespace RASTERVORONOIPACKING;

#define ZOOMNEIGHBORHOOD 3

// TOREMOVE
QString printSequenceWithChanges(QVector<int> seq, QVector<int> seqChanges) {
	QString ans;
	QTextStream deb(&ans);
	//QDebug deb = qDebug();
	deb << "{";
	//deb.nospace();
	foreach(int id, seq) {		
		if (std::find(seqChanges.begin(), seqChanges.end(), id) != seqChanges.end()) deb << id << "'";
		else deb << id;
		if (id != seq[seq.count()-1]) deb << ", ";
	}
	QChar ch1, ch2; deb >> ch1 >> ch2;
	deb << "}";
	return ans;
}

void RasterStripPackingSolver::setProblem(std::shared_ptr<RasterPackingProblem> _problem, bool isZoomedProblem) {
    if(isZoomedProblem) {
        this->zoomedProblem = _problem;
        return;
    }
    this->originalProblem = _problem;
    //this->currentProblem = this->originalProblem;

    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
        for(uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
            std::shared_ptr<TotalOverlapMap> curMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(originalProblem->getIfps()->getRasterNoFitPolygon(-1,-1,originalProblem->getItemType(itemId),angle)));			
			curMap->initCacheInfo(originalProblem->count());// TEST
            maps.addOverlapMap(itemId,angle,curMap);
            // FIXME: Delete innerift polygons as they are used to release memomry
        }
    }

    glsWeights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(originalProblem->count()));

    currentWidth = this->originalProblem->getContainerWidth();
    initialWidth = currentWidth;
}

// Basic Functions
void RasterStripPackingSolver::generateRandomSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<RasterPackingProblem> problem = params.isDoubleResolution() ? zoomedProblem : originalProblem;

    for(int i =0; i < originalProblem->count(); i++)  {
        // Shuffle angle
        int totalAngles = originalProblem->getItem(i)->getAngleCount();
        int rnd_angle = 0;
        if(totalAngles != 0) {
            rnd_angle = qrand() % ((totalAngles -1 + 1) - 0) + 0;
            solution.setOrientation(i, rnd_angle);
        }

        // Shuffle position
		std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(-1, -1, originalProblem->getItemType(i), rnd_angle);
		int newIfpWidth = ifp->width() - qRound(problem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
        int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
        int maxX = minX + newIfpWidth - 1;
        int maxY = minY + ifp->height() - 1;

        int rnd_x =  qrand() % ((maxX + 1) - minX) + minX;
        int rnd_y =  qrand() % ((maxY + 1) - minY) + minY;
        solution.setPosition(i, QPoint(rnd_x, rnd_y));
    }
}

// --> Generate initial solution using the bottom left heuristic and resize the container accordingly
void RasterStripPackingSolver::generateBottomLeftSolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<RasterPackingProblem> problem = params.isDoubleResolution() ? zoomedProblem : originalProblem;

	QVector<int> sequence;
	for (int i = 0; i < problem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	int layoutLength = 0;
	for (int k = 0; k < problem->count(); k++) {
		int shuffledId = sequence[k];
		int minMaxItemX;
		// Find left bottom placement for item
		for (unsigned int angle = 0; angle < problem->getItem(shuffledId)->getAngleCount(); angle++) {
			// Get IFP bounding box
			int  minIfpX, minIfpY, maxIfpX, maxIfpY;
			getIfpBoundingBox(shuffledId, angle, minIfpX, minIfpY, maxIfpX, maxIfpY, problem);
			QPoint curPos(minIfpX, minIfpY);
			while (1) { // FIXME: Infinite loop?
				//qreal overlap = getItemPartialOverlap(sequence, k, curPos, angle, solution, problem);
				//if (qFuzzyCompare(1.0 + 0.0, 1.0 + overlap)) break;
				if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution, problem))
					break;
				// Get next position
				curPos.setY(curPos.y() + 1); if (curPos.y() > maxIfpY) { curPos.setY(minIfpY); curPos.setX(curPos.x() + 1); }
			}
			// Check minimum X coordinate
			int maxItemX = getItemMaxX(curPos.x(), angle, shuffledId, problem);
			if (angle == 0 || maxItemX < minMaxItemX) {
				minMaxItemX = maxItemX;
				solution.setPosition(shuffledId, curPos); solution.setOrientation(shuffledId, angle);
			}
		}
		if (minMaxItemX > layoutLength) layoutLength = minMaxItemX;
	}
	if (params.isDoubleResolution()) layoutLength = qCeil((layoutLength / zoomedProblem->getScale()) * originalProblem->getScale());
	setContainerWidth(layoutLength, solution, params);
}

void RasterStripPackingSolver::getIfpBoundingBox(int itemId, int angle, int &bottomLeftX, int &bottomLeftY, int &topRightX, int &topRightY, std::shared_ptr<RasterPackingProblem> problem) {
	std::shared_ptr<RasterNoFitPolygon> ifp = problem->getIfps()->getRasterNoFitPolygon(-1, -1, problem->getItemType(itemId), angle);
	bottomLeftX = -ifp->getOriginX();
	bottomLeftY = -ifp->getOriginY();
	topRightX = bottomLeftX + ifp->width() - 1;
	topRightY = bottomLeftY + ifp->height() - 1;
}

// Determine item overlap for a subset of fixed items
qreal RasterStripPackingSolver::getItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem) {
	qreal overlap = 0;
	int itemId = sequence[itemSequencePos];
	for (int i = 0; i < itemSequencePos; i++) {
		int curFixedItemId = sequence[i];
		overlap += getDistanceValue(itemId, itemPos, itemAngle, curFixedItemId, solution.getPosition(curFixedItemId), solution.getOrientation(curFixedItemId), problem);
	}
	return overlap;
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
	std::shared_ptr<RasterPackingProblem> problem = params.isDoubleResolution() ? zoomedProblem : originalProblem;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		totalOverlap += getItemTotalOverlap(itemId, solution, problem);
    }
    return totalOverlap;
}

void RasterStripPackingSolver::updateItemCacheInfo(int itemId, QPoint oldPos, int oldAngle, bool useGlsWeights) {	
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		for (uint curAngle = 0; curAngle < originalProblem->getItem(i)->getAngleCount(); curAngle++) {
			std::shared_ptr<TotalOverlapMap> curPieceMap = maps.getOverlapMap(i, curAngle);
			if (useGlsWeights) curPieceMap->getCacheInfo(itemId)->cacheOldPlacement(oldPos, oldAngle, glsWeights->getWeight(itemId, i));
			else curPieceMap->getCacheInfo(itemId)->cacheOldPlacement(oldPos, oldAngle);
		}
	}
}

void RasterStripPackingSolver::updateItemCacheInfo(int itemId, QPoint oldPos, int oldAngle, RasterStripPackingParameters &params) {
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		for (uint curAngle = 0; curAngle < originalProblem->getItem(i)->getAngleCount(); curAngle++) {
			std::shared_ptr<TotalOverlapMap> curPieceMap = maps.getOverlapMap(i, curAngle);
			if (params.getHeuristic() == GLS) curPieceMap->getCacheInfo(itemId)->cacheOldPlacement(oldPos, oldAngle, glsWeights->getWeight(itemId, i));
			else curPieceMap->getCacheInfo(itemId)->cacheOldPlacement(oldPos, oldAngle);
		}
	}
}

void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
    newSolution = RasterPackingSolution(originalSolution.getNumItems());
    for(int i = 0; i < originalSolution.getNumItems(); i++) {
        newSolution.setOrientation(i, originalSolution.getOrientation(i));
        QPoint finePos = QPoint(qRound((qreal)originalSolution.getPosition(i).x() * scaleFactor), qRound((qreal)originalSolution.getPosition(i).y() * scaleFactor));
        newSolution.setPosition(i, finePos);
    }
}

void RasterStripPackingSolver::updateWeights(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	QVector<WeightIncrement> solutionOverlapValues;
	qreal maxOValue = 0;

	std::shared_ptr<RasterPackingProblem> problem = params.isDoubleResolution() ? zoomedProblem : originalProblem;

	// Determine pair overlap values
	for (int itemId1 = 0; itemId1 < originalProblem->count(); itemId1++)
	for (int itemId2 = 0; itemId2 < originalProblem->count(); itemId2++) {
		if (itemId1 == itemId2) continue;
		qreal curOValue = getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
			itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2), problem);
		if (curOValue != 0) {
			solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, (qreal)curOValue));
			if (curOValue > maxOValue) maxOValue = curOValue;
			// TEST
			for (uint curAngle = 0; curAngle < originalProblem->getItem(itemId1)->getAngleCount(); curAngle++) maps.getOverlapMap(itemId1, curAngle)->getCacheInfo(itemId2)->cacheOldPlacement(solution.getPosition(itemId2), solution.getOrientation(itemId2), glsWeights->getWeight(itemId1, itemId2));
			for (uint curAngle = 0; curAngle < originalProblem->getItem(itemId2)->getAngleCount(); curAngle++) maps.getOverlapMap(itemId2, curAngle)->getCacheInfo(itemId1)->cacheOldPlacement(solution.getPosition(itemId1), solution.getOrientation(itemId1), glsWeights->getWeight(itemId2, itemId1));
		}
	}

	// Divide vector by maximum
	std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value / maxOValue; });

	// Add to the current weight map
	glsWeights->updateWeights(solutionOverlapValues);
}

//  TODO: Update cache information!
void RasterStripPackingSolver::resetWeights() {
    glsWeights->reset(originalProblem->count());
	for (int itemId = 0; itemId < originalProblem->count(); itemId++)
		for (uint curAngle = 0; curAngle < originalProblem->getItem(itemId)->getAngleCount(); curAngle++)
			maps.getOverlapMap(itemId, curAngle)->resetCacheInfo(true);
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value, PositionChoice placementHeuristic) {
	float fvalue = value;
	QPoint minRelativePos = map->getMinimum(fvalue, placementHeuristic);
    return minRelativePos - map->getReferencePoint();
}

// TEST
qreal RasterStripPackingSolver::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, std::shared_ptr<RasterPackingProblem> problem, bool useGlsWeights) {
    qreal totalOverlap = 0;
    if(!useGlsWeights)
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
			totalOverlap += getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i), problem);
        }
    else
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
			totalOverlap += glsWeights->getWeight(itemId, i)*getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i), problem);
        }
//    qDebug() << pos << totalOverlap;
    return totalOverlap;
}

// --> Retrieve a rectangular area of the total overlap map
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, bool useGlsWeights) {
    // FIXME: Check if zoomed area is inside the innerfit polygon
    std::shared_ptr<RasterNoFitPolygon> curIfp = this->zoomedProblem->getIfps()->getRasterNoFitPolygon(-1,-1,this->originalProblem->getItemType(itemId),orientation);
    int newIfpWidth = curIfp->width() - qRound(this->zoomedProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
    int bottomleftX = pos.x() -  width/2 < -curIfp->getOriginX() ? -curIfp->getOriginX() : pos.x() -  width/2;
    int bottomleftY = pos.y() - height/2 < -curIfp->getOriginY() ? -curIfp->getOriginY() : pos.y() - height/2;
    int topRightX   = pos.x() +  width/2 > -curIfp->getOriginX() + newIfpWidth - 1 ? -curIfp->getOriginX() + newIfpWidth - 1: pos.x() +  width/2;
    int topRightY   = pos.y() +  height/2 > -curIfp->getOriginY() + curIfp->height() - 1 ? -curIfp->getOriginY() + curIfp->height() - 1: pos.y() +  height/2;
    width = topRightX - bottomleftX + 1;
    height = topRightY - bottomleftY + 1;
    std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(width, height));
    curZoomedMap->setReferencePoint(-QPoint(bottomleftX, bottomleftY));

    //switchProblem(true);
    for(int j = bottomleftY; j <= topRightY; j++)
        for(int i = bottomleftX; i <= topRightX; i++)
            curZoomedMap->setValue(QPoint(i, j), getTotalOverlapMapSingleValue(itemId, orientation, QPoint(i,j), solution, this->zoomedProblem, useGlsWeights));
    //switchProblem(false);

    return curZoomedMap;
}

// --> Change container size
void RasterStripPackingSolver::setContainerWidth(int pixelWidth, RasterStripPackingParameters &params) {
    int deltaPixels = this->currentWidth - pixelWidth;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++)
        for(uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
            std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
            curMap->shrink(deltaPixels);
			size_t curMapMem = curMap->getWidth()*curMap->getHeight()*sizeof(qreal);
			if (params.isGpuProcessing() && deltaPixels < 0) CUDAPACKING::reallocDeviceMaxIfp(curMapMem);
        }

    currentWidth = pixelWidth;
}

void RasterStripPackingSolver::setContainerWidth(int pixelWidth, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	setContainerWidth(pixelWidth, params);
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, solution.getOrientation(itemId));
		QPoint curItemPos = solution.getPosition(itemId);
		int maxPositionX = -curMap->getReferencePoint().x() + curMap->getWidth() - 1;
		if (params.isDoubleResolution()) maxPositionX = qRound((qreal)maxPositionX / originalProblem->getScale() * zoomedProblem->getScale());
		if (curItemPos.x() > maxPositionX) {
			// Translate to minimum overlap position. Test all orientations!
			qreal minValue; QPoint minPos; int minAngle = 0;
			minPos = getMinimumOverlapPosition(itemId, 0, solution, minValue, params);
			for (uint curAngle = 1; curAngle < originalProblem->getItem(itemId)->getAngleCount(); curAngle++) {
				qreal curValue; QPoint curPos;
				curPos = getMinimumOverlapPosition(itemId, curAngle, solution, curValue, params);
				if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
			}
			solution.setOrientation(itemId, minAngle);
			solution.setPosition(itemId, minPos);
		}
	}
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
	std::shared_ptr<RasterPackingProblem> problem = params.isDoubleResolution() ? zoomedProblem : originalProblem;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		qreal itemOverlap = getItemTotalOverlap(itemId, solution, problem);
        individualOverlaps.append(itemOverlap);
        totalOverlap += itemOverlap;
    }
    return totalOverlap;
}

// --> Return total overlap map for a given item using GPU
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMapGPU(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();

	// --> Converting parameter for cuda input

	// Inner fit polygon data conversion
	int ifpWidth, ifpHeight, ifpX, ifpY;
	ifpWidth = currrentPieceMap->getWidth(); ifpHeight = currrentPieceMap->getHeight();
	ifpX = currrentPieceMap->getReferencePoint().x(); ifpY = currrentPieceMap->getReferencePoint().y();

	// Solution data conversion
	int *placementsx, *placementsy, *angles; float *weights;
	placementsx = (int*)malloc(originalProblem->count()*sizeof(int));
	placementsy = (int*)malloc(originalProblem->count()*sizeof(int));
	angles = (int*)malloc(originalProblem->count()*sizeof(int));
	weights = (float*)malloc(originalProblem->count()*sizeof(float));
	for (int k = 0; k < originalProblem->count(); k++) {
		placementsx[k] = solution.getPosition(k).x();
		placementsy[k] = solution.getPosition(k).y();
		angles[k] = solution.getOrientation(k);
		if (useGlsWeights && k != itemId) weights[k] = glsWeights->getWeight(itemId, k);
	}

	// --> Determine the overlap map
	float *overlapMapRawData = CUDAPACKING::getcuOverlapMap(itemId, orientation, originalProblem->count(), 4, ifpWidth, ifpHeight, ifpX, ifpY, placementsx, placementsy, angles, weights, useGlsWeights);
	currrentPieceMap->setData(overlapMapRawData);

	// Free pointers
	free(placementsx);
	free(placementsy);
	free(angles);
	free(weights);

	return currrentPieceMap;
}

void RasterStripPackingSolver::printCompleteCacheInfo(int itemId, int orientation, bool useGlsWeights) {
	qDebug() << "Cache information of item" << itemId;
	std::shared_ptr<TotalOverlapMap> curPieceMap = maps.getOverlapMap(itemId, orientation);
	for (int i = 0; i < originalProblem->count(); i++) {
		if (curPieceMap->getCacheInfo(i)->changedPlacement())
		if (useGlsWeights) qDebug() << "Item" << i << "moved. Original position" << curPieceMap->getCacheInfo(i)->getPosition() << ". Original weight:" << curPieceMap->getCacheInfo(i)->getWeight();
			else qDebug() << "Item" << i << "moved. Original position" << curPieceMap->getCacheInfo(i)->getPosition();
	}
}

void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	if (!params.isDoubleResolution()) performLocalSearchSingleResolution(solution, params);
	else performLocalSearchDoubleResolution(solution, params);
}

void RasterStripPackingSolver::performLocalSearchSingleResolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
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
		if (params.isCacheMaps() && (minPos != solution.getPosition(shuffledId) || minAngle != solution.getOrientation(shuffledId)))
			updateItemCacheInfo(shuffledId, solution.getPosition(shuffledId), solution.getOrientation(shuffledId), params);
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

void RasterStripPackingSolver::performLocalSearchDoubleResolution(RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	// Solution must be given in finer scale
	RasterPackingSolution roughSolution;
	qreal zoomFactor = this->zoomedProblem->getScale()/this->originalProblem->getScale();
	int zoomSquareSize = ZOOMNEIGHBORHOOD*qRound(this->zoomedProblem->getScale() / this->originalProblem->getScale());
	
	QVector<int> sequence;
	for(int i = 0; i < originalProblem->count() ; i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	
	for(int i =0; i < originalProblem->count(); i++) {
		getScaledSolution(solution, roughSolution, 1.0/zoomFactor);
		int shuffledId = sequence[i];
		if (qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution, this->zoomedProblem))) continue;
		qreal minValue; QPoint minPos; int minAngle = 0;
		minPos = getMinimumOverlapPosition(shuffledId, minAngle, roughSolution, minValue, params);
		minPos = getZoomedMinimumOverlapPosition(shuffledId, minAngle, zoomFactor*minPos, zoomSquareSize, zoomSquareSize, solution, minValue, params);
		for(uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
			qreal curValue; QPoint curPos;
			curPos = getMinimumOverlapPosition(shuffledId, curAngle, roughSolution, curValue, params);
			curPos = getZoomedMinimumOverlapPosition(shuffledId, curAngle, zoomFactor*curPos, zoomSquareSize, zoomSquareSize, solution, curValue, params);
			if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
		}
		if (params.isCacheMaps() && (minPos != solution.getPosition(shuffledId) || minAngle != solution.getOrientation(shuffledId)))
			updateItemCacheInfo(shuffledId, solution.getPosition(shuffledId), solution.getOrientation(shuffledId), params);
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

QPoint RasterStripPackingSolver::getZoomedMinimumOverlapPosition(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> map = getRectTotalOverlapMap(itemId, orientation, pos, width, height, solution, params.getHeuristic() == GLS);
	//float fvalue = value;
	float fvalue;
	QPoint minRelativePos = map->getMinimum(fvalue);
	value = fvalue;
	return minRelativePos - map->getReferencePoint();
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params) {
	if (params.isCacheMaps() && (params.isGpuProcessing() || params.isDoubleResolution()))
		qWarning() << "Cannot use GPU or multiresolution with map cache. Ignoring using map cache option.";

	if (params.isGpuProcessing()) return getMinimumOverlapPositionGPU(itemId, orientation, solution, value, params);

	std::shared_ptr<TotalOverlapMap> map = getTotalOverlapMapSerial(itemId, orientation, solution, params);
	//float fvalue = value;
	float fvalue;
	QPoint minRelativePos = map->getMinimum(fvalue, params.getPlacementCriteria());
	value = fvalue;
	return minRelativePos - map->getReferencePoint();
}

QPoint RasterStripPackingSolver::getMinimumOverlapPositionGPU(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, RasterStripPackingParameters &params) {
	int minx, miny;
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	currrentPieceMap->reset();

	// --> Converting parameter for cuda input

	// Inner fit polygon data conversion
	int ifpWidth, ifpHeight, ifpX, ifpY;
	ifpWidth = currrentPieceMap->getWidth(); ifpHeight = currrentPieceMap->getHeight();
	ifpX = currrentPieceMap->getReferencePoint().x(); ifpY = currrentPieceMap->getReferencePoint().y();

	// Solution data conversion
	int *placementsx, *placementsy, *angles; float *weights;
	placementsx = (int*)malloc(originalProblem->count()*sizeof(int));
	placementsy = (int*)malloc(originalProblem->count()*sizeof(int));
	angles = (int*)malloc(originalProblem->count()*sizeof(int));
	weights = (float*)malloc(originalProblem->count()*sizeof(float));
	for (int k = 0; k < originalProblem->count(); k++) {
		placementsx[k] = solution.getPosition(k).x();
		placementsy[k] = solution.getPosition(k).y();
		angles[k] = solution.getOrientation(k);
		if (params.getHeuristic() == GLS && k != itemId) weights[k] = glsWeights->getWeight(itemId, k);
	}

	// --> Determine the overlap map	 and minimum overlap placement
	QPoint minPos;
	if (params.getPlacementCriteria() == BOTTOMLEFT_POS) { // FIXME: GPU minimum search is not really bottom left but fixed random
		value = CUDAPACKING::getcuMinimumOverlap(itemId, orientation, originalProblem->count(), 4, ifpWidth, ifpHeight, ifpX, ifpY, placementsx, placementsy, angles, weights, minx, miny, params.getHeuristic() == GLS);
		minPos = QPoint(minx, miny);
	}
	else { // Determine overlap map on GPU and minimum value and position on CPU (using random placement heuristic)
		float *overlapMapRawData = CUDAPACKING::getcuOverlapMap(itemId, orientation, originalProblem->count(), 4, ifpWidth, ifpHeight, ifpX, ifpY, placementsx, placementsy, angles, weights, params.getHeuristic() == GLS);
		currrentPieceMap->setData(overlapMapRawData);
		float fvalue; QPoint minRelativePos = currrentPieceMap->getMinimum(fvalue, params.getPlacementCriteria()); value = fvalue;
		minPos =  minRelativePos - currrentPieceMap->getReferencePoint();
	}

	// Free pointers
	free(placementsx);
	free(placementsy);
	free(angles);
	free(weights);

	return minPos;
}

std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMapSerial(int itemId, int orientation, RasterPackingSolution &solution, RasterStripPackingParameters &params) {
	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	if (!params.isCacheMaps()) {
		if (params.getHeuristic() == NONE)  getTotalOverlapMapSerialNoCacheNoWeight(currrentPieceMap, itemId, orientation, solution);
		if (params.getHeuristic() == GLS)  getTotalOverlapMapSerialNoCacheWeight(currrentPieceMap, itemId, orientation, solution);
	}
	else {
		if (params.getHeuristic() == NONE)  getTotalOverlapMapSerialCacheNoWeight(currrentPieceMap, itemId, orientation, solution);
		if (params.getHeuristic() == GLS)  getTotalOverlapMapSerialCacheWeight(currrentPieceMap, itemId, orientation, solution);
	}

	currrentPieceMap->resetCacheInfo(); // TEST
	return currrentPieceMap;
}

void RasterStripPackingSolver::getTotalOverlapMapSerialNoCacheNoWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution) {
	map->reset();
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i));
	}
}

void RasterStripPackingSolver::getTotalOverlapMapSerialNoCacheWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution) {
	map->reset();
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
}

void RasterStripPackingSolver::getTotalOverlapMapSerialCacheNoWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution) {
	if (2 * map->getCacheCount() >= originalProblem->count()) {
		getTotalOverlapMapSerialNoCacheNoWeight(map, itemId, orientation, solution);
		return;
	}
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		std::shared_ptr<CachePlacementInfo> curCacheInfo = map->getCacheInfo(i);
		if (!curCacheInfo->changedPlacement()) continue;
		// Remove item's nfp from older position
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), curCacheInfo->getOrientation(), originalProblem->getItemType(itemId), orientation), curCacheInfo->getPosition(), -1.0);
		// Insert into new position
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i));
	}
}

void RasterStripPackingSolver::getTotalOverlapMapSerialCacheWeight(std::shared_ptr<TotalOverlapMap> map, int itemId, int orientation, RasterPackingSolution &solution) {
	if (2 * map->getCacheCount() >= originalProblem->count()) {
		getTotalOverlapMapSerialNoCacheWeight(map, itemId, orientation, solution);
		return;
	}
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		std::shared_ptr<CachePlacementInfo> curCacheInfo = map->getCacheInfo(i);
		if (!curCacheInfo->changedPlacement()) continue;
		// Remove item's nfp from older position
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), curCacheInfo->getOrientation(), originalProblem->getItemType(itemId), orientation), curCacheInfo->getPosition(), -curCacheInfo->getWeight());
		// Insert into new position
		map->addVoronoi(originalProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation), solution.getPosition(i), glsWeights->getWeight(itemId, i));
	}
}