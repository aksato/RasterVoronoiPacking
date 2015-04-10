#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>
#include "../cuda/gpuinfo.h"

using namespace RASTERVORONOIPACKING;

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
    this->currentProblem = this->originalProblem;

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
void RasterStripPackingSolver::generateRandomSolution(RasterPackingSolution &solution) {
    for(int i =0; i < originalProblem->count(); i++)  {
        // Shuffle angle
        int totalAngles = originalProblem->getItem(i)->getAngleCount();
        int rnd_angle = 0;
        if(totalAngles != 0) {
            rnd_angle = qrand() % ((totalAngles -1 + 1) - 0) + 0;
            solution.setOrientation(i, rnd_angle);
        }

        // Shuffle position
        std::shared_ptr<RasterNoFitPolygon> ifp = currentProblem->getIfps()->getRasterNoFitPolygon(-1,-1,originalProblem->getItemType(i),rnd_angle);
//        int newIfpWidth = ifp->getImage().width() - qRound(this->currentProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
        int newIfpWidth = ifp->width() - qRound(this->currentProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
        int minX = -ifp->getOriginX(); int minY = -ifp->getOriginY();
//        int maxX = minX + ifp->getImage().width() - 1;
        int maxX = minX + newIfpWidth - 1;
//        int maxY = minY + ifp->getImage().height() - 1;
        int maxY = minY + ifp->height() - 1;

        int rnd_x =  qrand() % ((maxX + 1) - minX) + minX;
        int rnd_y =  qrand() % ((maxY + 1) - minY) + minY;
        solution.setPosition(i, QPoint(rnd_x, rnd_y));
    }
}

// --> Get layout overlap (sum of individual overlap values)
qreal RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution) {
    qreal totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
        totalOverlap += getItemTotalOverlap(itemId, solution);
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

// --> Local search
void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution, bool useGlsWeights, bool cacheInfo) {
    QVector<int> sequence, changedSequence;
    for(int i = 0; i < originalProblem->count() ; i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
    for(int i =0; i < originalProblem->count(); i++) {
        int shuffledId = sequence[i];
        if(qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution))) continue;
        qreal minValue; QPoint minPos; int minAngle = 0;
        minPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, minAngle, solution, useGlsWeights), minValue);
        for(uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
            qreal curValue; QPoint curPos;
            curPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, curAngle, solution, useGlsWeights), curValue);
            if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
        }
		if (minPos == solution.getPosition(shuffledId) && minAngle == solution.getOrientation(shuffledId)) continue;
		if (cacheInfo) updateItemCacheInfo(shuffledId, solution.getPosition(shuffledId), solution.getOrientation(shuffledId), useGlsWeights); changedSequence.append(shuffledId); // TEST
        solution.setOrientation(shuffledId, minAngle);
        solution.setPosition(shuffledId, minPos);
    }
	//qDebug() << "Placement Sequence:" << qPrintable(printSequenceWithChanges(sequence, changedSequence)); //sequence; qDebug() << "Changed Sequence:" << changedSequence; // TOREMOVE
}

// --> Local search
void RasterStripPackingSolver::performLocalSearchGPU(RasterPackingSolution &solution, bool useGlsWeights) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());

	for (int i = 0; i < originalProblem->count(); i++) {
		int shuffledId = sequence[i];
		if (qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution))) continue;
		qreal minValue; QPoint minPos; int minAngle = 0;
		minPos = getMinimumOverlapPositionGPU(shuffledId, minAngle, solution, minValue, useGlsWeights);
		for (uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
			qreal curValue; QPoint curPos;
			curPos = getMinimumOverlapPositionGPU(shuffledId, curAngle, solution, curValue, useGlsWeights);
			if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
		}
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

// --> Switch between original and zoomed problems
void RasterStripPackingSolver::switchProblem(bool zoomedProblem) {
    if(zoomedProblem) this->currentProblem = this->zoomedProblem;
    else this->currentProblem = this->originalProblem;
}

void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
    newSolution = RasterPackingSolution(originalSolution.getNumItems());
    for(int i = 0; i < originalSolution.getNumItems(); i++) {
        newSolution.setOrientation(i, originalSolution.getOrientation(i));
        QPoint finePos = QPoint(qRound((qreal)originalSolution.getPosition(i).x() * scaleFactor), qRound((qreal)originalSolution.getPosition(i).y() * scaleFactor));
        newSolution.setPosition(i, finePos);
    }
}

// --> Local search with zoomed approach
void RasterStripPackingSolver::performTwoLevelLocalSearch(RasterPackingSolution &zoomedSolution, bool useGlsWeights, int neighboordScale) {
    // Solution must be given in finer scale
    RasterPackingSolution roughSolution;
    qreal zoomFactor = this->zoomedProblem->getScale()/this->originalProblem->getScale();
    int zoomSquareSize = neighboordScale*qRound(this->zoomedProblem->getScale()/this->originalProblem->getScale());


    QVector<int> sequence;
    for(int i = 0; i < originalProblem->count() ; i++) sequence.append(i);
    std::random_shuffle(sequence.begin(), sequence.end());

    for(int i =0; i < originalProblem->count(); i++) {
      getScaledSolution(zoomedSolution, roughSolution, 1.0/zoomFactor);

      int shuffledId = sequence[i];
      switchProblem(true); if(qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, zoomedSolution))) continue;
      qreal minValue; QPoint minPos; int minAngle = 0;
      switchProblem(false); minPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, minAngle, roughSolution, useGlsWeights), minValue);
      switchProblem(true); minPos = getMinimumOverlapPosition(getRectTotalOverlapMap(shuffledId, minAngle, zoomFactor*minPos, zoomSquareSize, zoomSquareSize, zoomedSolution, useGlsWeights), minValue);

      for(uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
          qreal curValue; QPoint curPos;
          switchProblem(false); curPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, curAngle, roughSolution, useGlsWeights), curValue);
          switchProblem(true); curPos = getMinimumOverlapPosition(getRectTotalOverlapMap(shuffledId, curAngle, zoomFactor*curPos, zoomSquareSize, zoomSquareSize, zoomedSolution, useGlsWeights), curValue);
//          curPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, curAngle, solution, useGlsWeights), curValue);
          if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
      }
      zoomedSolution.setOrientation(shuffledId, minAngle);
      zoomedSolution.setPosition(shuffledId, minPos);
    }

}

// --> GLS weights functions.
void RasterStripPackingSolver::updateWeights(RasterPackingSolution &solution) {
    QVector<WeightIncrement> solutionOverlapValues;
    qreal maxOValue = 0;

    // Determine pair overlap values
    for(int itemId1 = 0; itemId1 < originalProblem->count(); itemId1++)
            for(int itemId2 = 0; itemId2 < originalProblem->count(); itemId2++) {
                if(itemId1 == itemId2) continue;
                qreal curOValue = getDistanceValue(itemId1, solution.getPosition(itemId1), solution.getOrientation(itemId1),
                                                   itemId2, solution.getPosition(itemId2), solution.getOrientation(itemId2));
                if(curOValue != 0) {
                    solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, (qreal)curOValue));
                    if(curOValue > maxOValue) maxOValue = curOValue;
					// TEST
					for (uint curAngle = 0; curAngle < originalProblem->getItem(itemId1)->getAngleCount(); curAngle++) maps.getOverlapMap(itemId1, curAngle)->getCacheInfo(itemId2)->cacheOldPlacement(solution.getPosition(itemId2), solution.getOrientation(itemId2), glsWeights->getWeight(itemId1, itemId2));
					for (uint curAngle = 0; curAngle < originalProblem->getItem(itemId2)->getAngleCount(); curAngle++) maps.getOverlapMap(itemId2, curAngle)->getCacheInfo(itemId1)->cacheOldPlacement(solution.getPosition(itemId1), solution.getOrientation(itemId1), glsWeights->getWeight(itemId2, itemId1));
                }
            }

    // Divide vector by maximum
    std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value/maxOValue;});

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

// --> Return total overlap map for a given item
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMap(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights) {
    if(currentProblem == zoomedProblem) {
        //Q_ASSERT_X(width-pixels > 0, "RasterStripPackingSolver::getTotalOverlapMap", "Tried to obtain total overlap map, which is not permitted.");
        return 0;
    }

    std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
    currrentPieceMap->reset();
    if(!useGlsWeights) {
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
            currrentPieceMap->addVoronoi(
                currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation),
                solution.getPosition(i)
            );
        }
    } else {
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
            currrentPieceMap->addVoronoi(
                currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation),
                solution.getPosition(i),
                glsWeights->getWeight(itemId, i)
            );
        }
    }
	currrentPieceMap->resetCacheInfo(); // TEST
    return currrentPieceMap;
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value) {
	float fvalue = value;
	QPoint minRelativePos = map->getMinimum(fvalue);
    return minRelativePos - map->getReferencePoint();
}

// TEST
qreal RasterStripPackingSolver::getTotalOverlapMapSingleValue(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, bool useGlsWeights) {
    qreal totalOverlap = 0;
    if(!useGlsWeights)
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
            totalOverlap += getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i));
        }
    else
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
            totalOverlap += glsWeights->getWeight(itemId, i)*getDistanceValue(itemId, pos, orientation, i, solution.getPosition(i), solution.getOrientation(i));
        }
//    qDebug() << pos << totalOverlap;
    return totalOverlap;
}

// --> Retrieve a rectangular area of the total overlap map
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getRectTotalOverlapMap(int itemId, int orientation, QPoint pos, int width, int height, RasterPackingSolution &solution, bool useGlsWeights) {
    // FIXME: Check if zoomed area is inside the innerfit polygon
    std::shared_ptr<RasterNoFitPolygon> curIfp = this->currentProblem->getIfps()->getRasterNoFitPolygon(-1,-1,this->originalProblem->getItemType(itemId),orientation);
//    int newIfpWidth = curIfp->getImage().width() - qRound(this->currentProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
    int newIfpWidth = curIfp->width() - qRound(this->currentProblem->getScale() * (qreal)(this->initialWidth - this->currentWidth) / this->originalProblem->getScale());
    int bottomleftX = pos.x() -  width/2 < -curIfp->getOriginX() ? -curIfp->getOriginX() : pos.x() -  width/2;
    int bottomleftY = pos.y() - height/2 < -curIfp->getOriginY() ? -curIfp->getOriginY() : pos.y() - height/2;
    int topRightX   = pos.x() +  width/2 > -curIfp->getOriginX() + newIfpWidth - 1 ? -curIfp->getOriginX() + newIfpWidth - 1: pos.x() +  width/2;
//    int topRightY   = pos.y() +  height/2 > -curIfp->getOriginY() + curIfp->getImage().height() - 1 ? -curIfp->getOriginY() + curIfp->getImage().height() - 1: pos.y() +  height/2;
    int topRightY   = pos.y() +  height/2 > -curIfp->getOriginY() + curIfp->height() - 1 ? -curIfp->getOriginY() + curIfp->height() - 1: pos.y() +  height/2;
    width = topRightX - bottomleftX + 1;
    height = topRightY - bottomleftY + 1;
//    qDebug() << pos << curIfp->getOrigin() << bottomleftX << bottomleftY << topRightX << topRightY << width << height;
    std::shared_ptr<TotalOverlapMap> curZoomedMap = std::shared_ptr<TotalOverlapMap>(new TotalOverlapMap(width, height));
    curZoomedMap->setReferencePoint(-QPoint(bottomleftX, bottomleftY));

    switchProblem(true);
//    qDebug() << curZoomedMap->getReferencePoint() << width << height;
    for(int j = bottomleftY; j <= topRightY; j++)
        for(int i = bottomleftX; i <= topRightX; i++)
            curZoomedMap->setValue(QPoint(i, j), getTotalOverlapMapSingleValue(itemId, orientation, QPoint(i,j), solution, useGlsWeights));
    switchProblem(false);

    return curZoomedMap;
}

// --> Change container size
void RasterStripPackingSolver::setContainerWidth(int pixelWidth) {
    int deltaPixels = this->currentWidth - pixelWidth;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++)
        for(uint angle = 0; angle < originalProblem->getItem(itemId)->getAngleCount(); angle++) {
            std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
            curMap->shrink(deltaPixels);
        }

    currentWidth = pixelWidth;
}

void RasterStripPackingSolver::setContainerWidth(int pixelWidth, RasterPackingSolution &solution) {
	setContainerWidth(pixelWidth);
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, solution.getOrientation(itemId));
		QPoint curItemPos = solution.getPosition(itemId);
		int maxPositionX = -curMap->getReferencePoint().x() + curMap->getWidth() - 1;
		if (curItemPos.x() > maxPositionX) solution.setPosition(itemId, QPoint(maxPositionX, curItemPos.y()));
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
qreal RasterStripPackingSolver::getDistanceValue(int itemId1, QPoint pos1, int orientation1, int itemId2, QPoint pos2, int orientation2) {
    std::shared_ptr<RasterNoFitPolygon> curNfp1Static2Orbiting, curNfp2Static1Orbiting;
    qreal value1Static2Orbiting, value2Static1Orbiting;
    bool feasible;

    curNfp1Static2Orbiting = currentProblem->getNfps()->getRasterNoFitPolygon(
        originalProblem->getItemType(itemId1), orientation1,
        originalProblem->getItemType(itemId2), orientation2);
    value1Static2Orbiting = getNfpValue(pos1, pos2, curNfp1Static2Orbiting, feasible);
    if(feasible) return 0.0;

    curNfp2Static1Orbiting = currentProblem->getNfps()->getRasterNoFitPolygon(
            originalProblem->getItemType(itemId2), orientation2,
            originalProblem->getItemType(itemId1), orientation1);
    value2Static1Orbiting = getNfpValue(pos2, pos1, curNfp2Static1Orbiting, feasible);
    if(feasible) return 0.0;

    return value1Static2Orbiting < value2Static1Orbiting ? value1Static2Orbiting : value2Static1Orbiting;
}

qreal RasterStripPackingSolver::getItemTotalOverlap(int itemId, RasterPackingSolution &solution) {
    qreal totalOverlap = 0;
    for(int i =0; i < originalProblem->count(); i++) {
        if(i == itemId) continue;
        totalOverlap += getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId),
                                              i, solution.getPosition(i), solution.getOrientation(i));
    }
    return totalOverlap;
}

qreal RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution, QVector<qreal> &individualOverlaps) {
    qreal totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
        qreal itemOverlap = getItemTotalOverlap(itemId, solution);
        individualOverlaps.append(itemOverlap);
        totalOverlap += itemOverlap;
    }
    return totalOverlap;
}

// --> Return total overlap map for a given item using GPU
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMapGPU(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights) {
	if (currentProblem == zoomedProblem) {
		//Q_ASSERT_X(width-pixels > 0, "RasterStripPackingSolver::getTotalOverlapMap", "Tried to obtain total overlap map, which is not permitted.");
		return 0;
	}

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

QPoint RasterStripPackingSolver::getMinimumOverlapPositionGPU(int itemId, int orientation, RasterPackingSolution &solution, qreal &value, bool useGlsWeights) {
	if (currentProblem == zoomedProblem) {
		//Q_ASSERT_X(width-pixels > 0, "RasterStripPackingSolver::getTotalOverlapMap", "Tried to obtain total overlap map, which is not permitted.");
		return QPoint(0,0);
	}

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
		if (useGlsWeights && k != itemId) weights[k] = glsWeights->getWeight(itemId, k);
	}

	// --> Determine the overlap map	 and minimum overlap placement
	value = CUDAPACKING::getcuMinimumOverlap(itemId, orientation, originalProblem->count(), 4, ifpWidth, ifpHeight, ifpX, ifpY, placementsx, placementsy, angles, weights, minx, miny, useGlsWeights);

	// Free pointers
	free(placementsx);
	free(placementsy);
	free(angles);
	free(weights);

	return QPoint(minx, miny);
}

// Cached Overlap map determination. TODO: Use maps from pieces of same type
std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMapwithCache(int itemId, int orientation, RasterPackingSolution &solution, bool useGlsWeights) {
	if (currentProblem == zoomedProblem) {
		//Q_ASSERT_X(width-pixels > 0, "RasterStripPackingSolver::getTotalOverlapMap", "Tried to obtain total overlap map, which is not permitted.");
		return 0;
	}

	std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
	if (2 * currrentPieceMap->getCacheCount() >= originalProblem->count()) {
		//qDebug() << "Not using cache for item" << itemId << " with orientation" << orientation << ".";
		return getTotalOverlapMap(itemId, orientation, solution, useGlsWeights);
	}
	//qDebug() << "Using cache for item" << itemId << " with orientation" << orientation << ". Number of moved pieces" << currrentPieceMap->getCacheCount(); //printCompleteCacheInfo(itemId, orientation, useGlsWeights);
	if (!useGlsWeights) {
		for (int i = 0; i < originalProblem->count(); i++) {
			std::shared_ptr<CachePlacementInfo> curCacheInfo = currrentPieceMap->getCacheInfo(i);
			if (i == itemId || !curCacheInfo->changedPlacement()) continue;
			currrentPieceMap->addVoronoi(
				currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), curCacheInfo->getOrientation(), originalProblem->getItemType(itemId), orientation),
				curCacheInfo->getPosition(), -1.0
				);
			currrentPieceMap->addVoronoi(
				currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation),
				solution.getPosition(i)
				);
		}
	}
	else {
		for (int i = 0; i < originalProblem->count(); i++) {
			std::shared_ptr<CachePlacementInfo> curCacheInfo = currrentPieceMap->getCacheInfo(i);
			if (i == itemId || !curCacheInfo->changedPlacement()) continue;
			currrentPieceMap->addVoronoi(
				currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), curCacheInfo->getOrientation(), originalProblem->getItemType(itemId), orientation),
				curCacheInfo->getPosition(), -curCacheInfo->getWeight()
				);
			currrentPieceMap->addVoronoi(
				currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation),
				solution.getPosition(i),
				glsWeights->getWeight(itemId, i)
				);
		}
	}
	currrentPieceMap->resetCacheInfo();
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

void RasterStripPackingSolver::performLocalSearchwithCache(RasterPackingSolution &solution, bool useGlsWeights) {
	QVector<int> sequence, changedSequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	for (int i = 0; i < originalProblem->count(); i++) {
		int shuffledId = sequence[i];
		if (qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution))) continue;
		qreal minValue; QPoint minPos; int minAngle = 0;
		minPos = getMinimumOverlapPosition(getTotalOverlapMapwithCache(shuffledId, minAngle, solution, useGlsWeights), minValue);
		for (uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
			qreal curValue; QPoint curPos;
			curPos = getMinimumOverlapPosition(getTotalOverlapMapwithCache(shuffledId, curAngle, solution, useGlsWeights), curValue);
			if (curValue < minValue) { minValue = curValue; minPos = curPos; minAngle = curAngle; }
		}
		if (minPos == solution.getPosition(shuffledId) && minAngle == solution.getOrientation(shuffledId)) continue;
		updateItemCacheInfo(shuffledId, solution.getPosition(shuffledId), solution.getOrientation(shuffledId), useGlsWeights); changedSequence.append(shuffledId); // TEST
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
	//qDebug() << "Placement Sequence:" << sequence; qDebug() << "Changed Sequence:" << changedSequence; // TOREMOVE
}