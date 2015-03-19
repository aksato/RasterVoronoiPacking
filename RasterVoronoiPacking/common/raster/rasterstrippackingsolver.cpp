#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>

//TOERASE
#include <Qfile>
#include <QTextStream>

using namespace RASTERVORONOIPACKING;

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

// --> Local search
void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution, bool useGlsWeights) {
    QVector<int> sequence;
    for(int i = 0; i < originalProblem->count() ; i++) sequence.append(i);
    std::random_shuffle(sequence.begin(), sequence.end());

    for(int i =0; i < originalProblem->count(); i++) {
        int shuffledId = sequence[i];
        if(qFuzzyCompare(1.0 + 0.0, 1.0 + getItemTotalOverlap(shuffledId, solution))) continue;
        qreal minValue; QPoint minPos; int minAngle = 0;
        minPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, minAngle, solution, useGlsWeights), minValue);
//        determineMap(shuffledId, solution, minAngle); minPos = determineMinPos(shuffledId, minAngle, minValue);
        for(uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
            qreal curValue; QPoint curPos;
            curPos = getMinimumOverlapPosition(getTotalOverlapMap(shuffledId, curAngle, solution, useGlsWeights), curValue);
//            determineMap(shuffledId, solution, curAngle); curPos = determineMinPos(shuffledId, curAngle, curValue);
            if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
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

// --> GLS weights functions
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
                }
            }

    // Divide vector by maximum
    std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value/maxOValue;});

    // Add to the current weight map
    glsWeights->updateWeights(solutionOverlapValues);
}

void RasterStripPackingSolver::resetWeights() {
    glsWeights->reset(originalProblem->count());
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

			// TOERASE
			currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation)->save("nfp" + QString::number(i) + ".txt");// TEST
			qDebug() << originalProblem->getItemType(i) << solution.getOrientation(i) << originalProblem->getItemType(itemId) << orientation << currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation)->width() << currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation)->height();
        }
    } else {
		// TOERASE
		QVector<qreal> weights;
        for(int i =0; i < originalProblem->count(); i++) {
            if(i == itemId) continue;
            currrentPieceMap->addVoronoi(
                currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation),
                solution.getPosition(i),
                glsWeights->getWeight(itemId, i)
            );

			// TOERASE
			weights.push_back(glsWeights->getWeight(itemId, i));
			currentProblem->getNfps()->getRasterNoFitPolygon(originalProblem->getItemType(i), solution.getOrientation(i), originalProblem->getItemType(itemId), orientation)->save("nfp" + QString::number(i) + ".txt");// TEST
        }
		// TOERASE
		QFile outfile("weights.txt");
		if (outfile.open(QFile::WriteOnly)) {
			QTextStream out(&outfile);
			for(QVector<qreal>::iterator it = weights.begin(); it != weights.end(); it++) out << QString::number(*it) << " ";
		}
		outfile.close();
    }
    return currrentPieceMap;
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(std::shared_ptr<TotalOverlapMap> map, qreal &value) {
    QPoint minRelativePos = map->getMinimum(value);
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

//void RasterStripPackingSolver::setZoomedProblem(std::shared_ptr<RasterPackingProblem> _zoomedProblem) {
//    this->zoomedProblem = _zoomedProblem;

//    int length = qRound(this->zoomedProblem->getScale()/this->problem->getScale());

//    for(int itemId = 0; itemId < problem->count(); itemId++) {
//        for(uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
//            std::shared_ptr<ZoomedTotalOverlapMap> curMap = std::shared_ptr<ZoomedTotalOverlapMap>(new ZoomedTotalOverlapMap(length,length));
//            zoomedMaps.addOverlapMap(itemId,angle,curMap);
//        }
//    }
////    zoomedMap = std::shared_ptr<ZoomedTotalOverlapMap>(new ZoomedTotalOverlapMap(length,length));
//}

//void RasterStripPackingSolver::setContainerWidth(int pixelWidth) {
//    int deltaPixels = this->currentWidth - pixelWidth;
//    for(int itemId = 0; itemId < problem->count(); itemId++)
//        for(uint angle = 0; angle < problem->getItem(itemId)->getAngleCount(); angle++) {
//            std::shared_ptr<TotalOverlapMap> curMap = maps.getOverlapMap(itemId, angle);
//            curMap->shrink(deltaPixels);
//        }

//    currentWidth = pixelWidth;
//}

//std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::determineMap(int itemId, RasterPackingSolution &solution, int orientation) {
//    std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
//    currrentPieceMap->reset();
//    for(int i =0; i < problem->count(); i++) {
//        if(i == itemId) continue;
//        currrentPieceMap->addVoronoi(
//            problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation),
//            solution.getPosition(i)
//        );
//    }
//    return currrentPieceMap;
//}

//QPoint RasterStripPackingSolver::determineMinPos(int itemId, int orientation, qreal &value) {
//    std::shared_ptr<TotalOverlapMap> currrentPieceMap;
//    currrentPieceMap = maps.getOverlapMap(itemId, orientation);
//    return currrentPieceMap->getMinimum(value);
//}

//std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalOverlapMap(int itemId, RasterPackingSolution &solution) {
//    return determineMap(itemId, solution);
//}

//void RasterStripPackingSolver::translateToMinimumOverlapPosition(int itemId, RasterPackingSolution &solution, qreal &value) {
//    QPoint newPos = determineMinPos(itemId, solution, value);
//    solution.setPosition(itemId, newPos);
//}

//// FIXME: Use function getTotalOverlap(int itemId, QPoint pos, RasterPackingSolution &solution)
//qreal RasterStripPackingSolver::getTotalOverlap(int itemId, RasterPackingSolution &solution) {
//    qreal totalOverlap = 0;
//    for(int i =0; i < problem->count(); i++) {
//        if(i == itemId) continue;
//        totalOverlap += getOverlap(itemId, i, solution);
//    }
//    return totalOverlap;
//}

//qreal RasterStripPackingSolver::getZoomedTotalOverlap(int itemId, RasterPackingSolution &solution) {
//    // Solution is in fine scale
//    qreal totalOverlap = 0;
//    for(int i =0; i < problem->count(); i++) {
//        if(i == itemId) continue;
//        totalOverlap += getZoomedOverlap(itemId, i, solution);
//    }
//    return totalOverlap;
//}

//qreal RasterStripPackingSolver::getZoomedTotalOverlap(int itemId, int orientation, QPoint pos, RasterPackingSolution &solution, qreal zoomFactor, bool weighted) {
//    qreal totalOverlap = 0;
//    for(int i =0; i < problem->count(); i++) {
//        if(i == itemId) continue;
//        if(weighted) totalOverlap += glsWeights->getWeight(itemId, i) * getZoomedOverlap(itemId, pos, orientation, i, solution, zoomFactor);
//        else totalOverlap += getZoomedOverlap(itemId, pos, orientation, i, solution, zoomFactor);
//    }
//    return totalOverlap;
//}

//void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution) {
//    QVector<int> sequence;
//    for(int i = 0; i < problem->count() ; i++) sequence.append(i);
//    std::random_shuffle(sequence.begin(), sequence.end());

//    for(int i =0; i < problem->count(); i++) {
//        int shuffledId = sequence[i];
//        if(qFuzzyCompare(1.0 + 0.0, 1.0 + getTotalOverlap(shuffledId, solution))) continue;
//        qreal minValue; QPoint minPos; int minAngle = 0;
//        determineMap(shuffledId, solution, minAngle); minPos = determineMinPos(shuffledId, minAngle, minValue);
//        for(uint curAngle = 1; curAngle < problem->getItem(shuffledId)->getAngleCount(); curAngle++) {
//            qreal curValue; QPoint curPos;
//            determineMap(shuffledId, solution, curAngle); curPos = determineMinPos(shuffledId, curAngle, curValue);
//            if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
//        }
//        solution.setOrientation(shuffledId, minAngle);
//        solution.setPosition(shuffledId, minPos);
//    }
//}

//void RasterStripPackingSolver::initGlsWeightedWeightValues() {
//    glsWeights->clear();
//    glsWeights->init(problem->count());
//}

//void RasterStripPackingSolver::resetGlsWeightedWeightValues() {
//    glsWeights->reset(problem->count());
//}

//// FIXME: Use function getOverlap(int itemId1, QPoint pos1, int itemId2, RasterPackingSolution &solution)
//qreal RasterStripPackingSolver::getOverlap(int itemId1, int itemId2, RasterPackingSolution &solution) {
////    QPoint pos1 = solution.getPosition(itemId1);
////    QPoint pos2 = solution.getPosition(itemId2);
//    QPoint pos2 = solution.getPosition(itemId1);
//    QPoint pos1 = solution.getPosition(itemId2);

//    std::shared_ptr<RasterNoFitPolygon> curNfp = problem->getNfps()->getRasterNoFitPolygon(
////        problem->getItemType(itemId1), solution.getOrientation(itemId1),
////        problem->getItemType(itemId2), solution.getOrientation(itemId2));
//        problem->getItemType(itemId2), solution.getOrientation(itemId2),
//        problem->getItemType(itemId1), solution.getOrientation(itemId1));

//    QPoint relPos = pos2 - pos1 + curNfp->getOrigin();
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
//    int indexValue = curNfp->getImage().pixelIndex(relPos);
//    if(indexValue == 0) return 0.0;
//    qreal value1 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    curNfp = problem->getNfps()->getRasterNoFitPolygon(
//            problem->getItemType(itemId1), solution.getOrientation(itemId1),
//            problem->getItemType(itemId2), solution.getOrientation(itemId2));
//    relPos = pos1 - pos2 + curNfp->getOrigin();
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
//    indexValue = curNfp->getImage().pixelIndex(relPos);
//    if(indexValue == 0) return 0.0;
//    qreal value2 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    return value1 < value2 ? value1 : value2;
//}

//qreal RasterStripPackingSolver::getZoomedOverlap(int itemId1, QPoint pos1, int orientation1, int itemId2, RasterPackingSolution &solution, qreal zoomFactor) {
//    // pos1 is in refined scale, solution is in rough scale
//    QPoint pos2 = pos1;
//    pos1 = solution.getPosition(itemId2);
//    pos1 = QPoint(qRound(pos1.x()*zoomFactor), qRound(pos1.y()*zoomFactor));

//    std::shared_ptr<RasterNoFitPolygon> curNfp = zoomedProblem->getNfps()->getRasterNoFitPolygon(
//        zoomedProblem->getItemType(itemId2), solution.getOrientation(itemId2),
//        zoomedProblem->getItemType(itemId1), orientation1);

//    QPoint relPos = pos2 - pos1 + curNfp->getOrigin();
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
////    return curNfp->getImage().pixelIndex(pos2 - pos1 + curNfp->getOrigin());
//    int indexValue = curNfp->getImage().pixelIndex(relPos);
//    if(indexValue == 0) return 0.0;
//    qreal value1 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    curNfp = zoomedProblem->getNfps()->getRasterNoFitPolygon(
//        zoomedProblem->getItemType(itemId1), orientation1,
//        zoomedProblem->getItemType(itemId2), solution.getOrientation(itemId2));
//    relPos = pos1 - pos2 + curNfp->getOrigin();
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
//    indexValue = curNfp->getImage().pixelIndex(relPos);
//    if(indexValue == 0) return 0.0;
//    qreal value2 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    return value1 < value2 ? value1 : value2;
//}


//qreal RasterStripPackingSolver::getZoomedOverlap(int itemId1, int itemId2, RasterPackingSolution &solution) {
//    // Items and solutions are in the same scale
//    QPoint pos2 = solution.getPosition(itemId1);
//    QPoint pos1 = solution.getPosition(itemId2);

//    std::shared_ptr<RasterNoFitPolygon> curNfp = zoomedProblem->getNfps()->getRasterNoFitPolygon(
//        zoomedProblem->getItemType(itemId2), solution.getOrientation(itemId2),
//        zoomedProblem->getItemType(itemId1), solution.getOrientation(itemId1));

//    QPoint relPos = pos2 - pos1 + curNfp->getOrigin();
////    qDebug() << itemId1 << itemId2 << relPos << pos2 << pos1 << curNfp->getOrigin() << curNfp->getImage().width() << curNfp->getImage().height() << curNfp->getImage().pixelIndex(relPos);
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
//    int indexValue = curNfp->getImage().pixelIndex(relPos);
////    curNfp->getImage().save("Teste.png");
//    if(indexValue == 0) return 0.0;
//    qreal value1 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    curNfp = zoomedProblem->getNfps()->getRasterNoFitPolygon(
//            zoomedProblem->getItemType(itemId1), solution.getOrientation(itemId1),
//            zoomedProblem->getItemType(itemId2), solution.getOrientation(itemId2));
//    relPos = pos1 - pos2 + curNfp->getOrigin();
//    if(relPos.x() < 0 || relPos.x() > curNfp->getImage().width()-1 || relPos.y() < 0 || relPos.y() > curNfp->getImage().height()-1) return 0;
//    indexValue = curNfp->getImage().pixelIndex(relPos);
//    if(indexValue == 0) return 0.0;
//    qreal value2 = 1.0 + (curNfp->getMaxD()-1.0)*((qreal)indexValue-1.0)/254.0;

//    return value1 < value2 ? value1 : value2;
//}

//void RasterStripPackingSolver::updateWeights(RasterPackingSolution &solution) {
//    QVector<WeightIncrement> solutionOverlapValues;
//    qreal maxOValue = 0;

//    // Determine pair overlap values
//    for(int itemId1 = 0; itemId1 < problem->count(); itemId1++)
//            for(int itemId2 = 0; itemId2 < problem->count(); itemId2++) {
//                if(itemId1 == itemId2) continue;
//                qreal curOValue = getOverlap(itemId1, itemId2, solution);
//                if(curOValue != 0) {
//                    solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, (qreal)curOValue));
//                    if(curOValue > maxOValue) maxOValue = curOValue;
//                }
//            }

//    // Divide vector by maximum
//    std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value/maxOValue;});

//    // Add to the current weight map
//    glsWeights->updateWeights(solutionOverlapValues);
//}

//std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::determineGlsWeightedMap(int itemId, RasterPackingSolution &solution, int orientation) {
//    std::shared_ptr<TotalOverlapMap> currrentPieceMap = maps.getOverlapMap(itemId, orientation);
//    currrentPieceMap->reset();
//    for(int i =0; i < problem->count(); i++) {
//        if(i == itemId) continue;
//        currrentPieceMap->addVoronoi(
//            problem->getNfps()->getRasterNoFitPolygon(problem->getItemType(i), solution.getOrientation(i), problem->getItemType(itemId), orientation),
//            solution.getPosition(i),
//            glsWeights->getWeight(itemId, i)
//        );
//    }
//    return currrentPieceMap;
//}

//std::shared_ptr<TotalOverlapMap> RasterStripPackingSolver::getTotalGlsWeightedOverlapMap(int itemId, RasterPackingSolution &solution) {
//    return determineGlsWeightedMap(itemId, solution);
//}

//qreal RasterStripPackingSolver::getTotalGlsWeightedOverlap(int itemId, RasterPackingSolution &solution) {
//    std::shared_ptr<TotalOverlapMap> currrentPieceMap = determineGlsWeightedMap(itemId, solution);
//    QPoint curPos = solution.getPosition(itemId);
//    return currrentPieceMap->getValue(curPos);
//}
//qreal RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution) {
//    qreal totalOverlap = 0;
//    for(int i = 0; i < problem->count(); i++)
//        totalOverlap += getTotalOverlap(i, solution);
//    return totalOverlap;
//}

//void RasterStripPackingSolver::performGlsWeightedLocalSearch(RasterPackingSolution &solution) {
//    QVector<int> sequence;
//    for(int i = 0; i < problem->count() ; i++) sequence.append(i);
//    std::random_shuffle(sequence.begin(), sequence.end());

//    for(int i =0; i < problem->count(); i++) {
//        int shuffledId = sequence[i];
//        if(qFuzzyCompare(1.0 + 0.0, 1.0 + getTotalOverlap(shuffledId, solution))) continue;
//        qreal minValue; QPoint minPos; int minAngle = 0;
//        determineGlsWeightedMap(shuffledId, solution, minAngle); minPos = determineMinPos(shuffledId, minAngle, minValue);
//        for(uint curAngle = 1; curAngle < problem->getItem(shuffledId)->getAngleCount(); curAngle++) {
//            qreal curValue; QPoint curPos;
//            determineGlsWeightedMap(shuffledId, solution, curAngle); curPos = determineMinPos(shuffledId, curAngle, curValue);
//            if(curValue < minValue) {minValue = curValue; minPos = curPos; minAngle = curAngle;}
//        }
//        solution.setOrientation(shuffledId, minAngle);
//        solution.setPosition(shuffledId, minPos);
//    }
//}

//void RasterStripPackingSolver::updateGlsWeightedWeightValues(RasterPackingSolution &solution) {
//    updateWeights(solution);
//}

//void RasterStripPackingSolver::getTotalZoomedOverlapMap(int itemId, QPoint nonZoomedpos, int orientation, RasterPackingSolution &solution) {
////    QPoint nonZoomedpos = solution.getPosition(itemId);
//    qreal zoomFactor = this->zoomedProblem->getScale()/this->problem->getScale();
//    QPoint bottomLeft = QPoint(qRound(nonZoomedpos.x()*zoomFactor), qRound(nonZoomedpos.y()*zoomFactor));

//    std::shared_ptr<ZoomedTotalOverlapMap> curZoomedMap = std::static_pointer_cast<ZoomedTotalOverlapMap>(zoomedMaps.getOverlapMap(itemId, orientation));
//    curZoomedMap->setOriginalCoords(bottomLeft);
//    curZoomedMap->setScaleFactor(zoomFactor);
//    curZoomedMap->reset();
//    int zoomI = 0;
//    int zoomJ = 0;

//    // FIXME: Check if zoomed area is inside the innerfit polygon
//    std::shared_ptr<RasterNoFitPolygon> curIfp = zoomedProblem->getIfps()->getRasterNoFitPolygon(-1,-1,problem->getItemType(itemId),orientation);
//    curZoomedMap->setValidArea(curIfp);
//    for(int j = bottomLeft.y(); j < bottomLeft.y()+curZoomedMap->getHeight(); j++) {
//        for(int i = bottomLeft.x(); i < bottomLeft.x()+curZoomedMap->getWidth(); i++) {
//            curZoomedMap->setPixel(zoomI, zoomJ, getZoomedTotalOverlap(itemId, orientation, QPoint(i,j), solution, zoomFactor) );
//            zoomI++;
//        }
//        zoomI = 0; zoomJ++;
//    }
//}

//QPoint RasterStripPackingSolver::determineMinZoomedPos(int itemId, int orientation, qreal &value) {
//    std::shared_ptr<ZoomedTotalOverlapMap> currrentZoomPieceMap;
//    currrentZoomPieceMap = std::static_pointer_cast<ZoomedTotalOverlapMap>(zoomedMaps.getOverlapMap(itemId, orientation));
//    return currrentZoomPieceMap->getMinimum(value);
//}

//void getScaledSolution(RasterPackingSolution &originalSolution, RasterPackingSolution &newSolution, qreal scaleFactor) {
//    newSolution = RasterPackingSolution(originalSolution.getNumItems());
//    for(int i = 0; i < originalSolution.getNumItems(); i++) {
//        newSolution.setOrientation(i, originalSolution.getOrientation(i));
//        QPoint finePos = QPoint(qFloor((qreal)originalSolution.getPosition(i).x() * scaleFactor), qFloor((qreal)originalSolution.getPosition(i).y() * scaleFactor));
//        newSolution.setPosition(i, finePos);
//    }
//}

//void RasterStripPackingSolver::translateToMinimumZoomedOverlapPosition(int itemId, RasterPackingSolution &solution, qreal &value) {
//    // Solution is given in a non zoomed scale
//    QPoint newRoughPos = determineMinPos(itemId, solution, value);
//    // Search zoomed space
//    getTotalZoomedOverlapMap(itemId, newRoughPos, solution.getOrientation(itemId), solution);
//    QPoint newPos = determineMinZoomedPos(itemId, solution, value);

//    // Solution is converted to a zoomed one
//    qreal zoomFactor = this->zoomedProblem->getScale()/this->problem->getScale();
//    RasterPackingSolution zoomedSolution; getScaledSolution(solution, zoomedSolution, zoomFactor);
//    zoomedSolution.setPosition(itemId, newPos);
//    solution = zoomedSolution; // FIXME: Inneficient?
//}

//qreal RasterStripPackingSolver::getGlobalZoomedOverlap(RasterPackingSolution &solution) {
//    // Input solution is in refined grid
////    qreal zoomFactor = this->zoomedProblem->getScale()/this->problem->getScale();
////    RasterPackingSolution roughSolution; getScaledSolution(solution, roughSolution, 1.0/zoomFactor);

//    qreal totalOverlap = 0;
//    for(int i = 0; i < problem->count(); i++)
////        totalOverlap += getZoomedTotalOverlap(i, solution.getOrientation(i), solution.getPosition(i), roughSolution, zoomFactor, false);
//        totalOverlap += getZoomedTotalOverlap(i, solution);

//    return totalOverlap;
//}

////qreal RasterStripPackingSolver::getGlobalZoomedOverlap(RasterPackingSolution &solution) {
////    // Input solution is in refined grid

////}

//void RasterStripPackingSolver::updateZoomedWeightValues(RasterPackingSolution &solution) {
//    // Input solution is in refined grid
//    qreal zoomFactor = this->zoomedProblem->getScale()/this->problem->getScale();
//    RasterPackingSolution roughSolution; getScaledSolution(solution, roughSolution, 1.0/zoomFactor);

//    QVector<WeightIncrement> solutionOverlapValues;
//    qreal maxOValue = 0;

//    // Determine pair overlap values
//    for(int itemId1 = 0; itemId1 < problem->count(); itemId1++)
//            for(int itemId2 = 0; itemId2 < problem->count(); itemId2++) {
//                if(itemId1 == itemId2) continue;
//                qreal curOValue = getZoomedOverlap(itemId1, itemId2, solution);
////                qreal curOValue2 = getZoomedOverlap(itemId2, itemId1, solution);
////                if(!qFuzzyCompare(1.0 + curOValue, 1.0 + curOValue2))
////                    qDebug() << "Hey!" << curOValue << curOValue2;
//                if(curOValue != 0) {
//                    solutionOverlapValues.append(WeightIncrement(itemId1, itemId2, curOValue));
//                    if(curOValue > maxOValue) maxOValue = curOValue;
//                }
//            }

//    // Divide vector by maximum
//    std::for_each(solutionOverlapValues.begin(), solutionOverlapValues.end(), [&maxOValue](WeightIncrement &curW){curW.value = curW.value/maxOValue;});

//    // Add to the current weight map
//    glsWeights->updateWeights(solutionOverlapValues);
//}

//void RasterStripPackingSolver::performZoomedLocalSearch(RasterPackingSolution &solution) {
//    // Input solution is in refined grid
//    qreal zoomFactor = this->zoomedProblem->getScale()/this->problem->getScale();

//    QVector<int> sequence;
//    for(int i = 0; i < problem->count() ; i++) sequence.append(i);
//    std::random_shuffle(sequence.begin(), sequence.end());

//    for(int i =0; i < problem->count(); i++) {
////    for(int i =0; i < 1; i++) {
//        int shuffledId = sequence[i];
//        RasterPackingSolution roughSolution; getScaledSolution(solution, roughSolution, 1.0/zoomFactor);

//        if(qFuzzyCompare(1.0 + 0.0, 1.0 + getZoomedTotalOverlap(shuffledId, solution.getOrientation(shuffledId), solution.getPosition(shuffledId), roughSolution, zoomFactor, false))) continue;
//        qreal minValue; QPoint minPos; int minAngle = 0;
//        determineGlsWeightedMap(shuffledId, roughSolution, minAngle); QPoint newRoughPos = determineMinPos(shuffledId, minAngle, minValue);
//        getTotalZoomedOverlapMap(shuffledId, newRoughPos, minAngle, roughSolution);
//        minPos = determineMinZoomedPos(shuffledId, minAngle, minValue);

//        for(uint curAngle = 1; curAngle < problem->getItem(shuffledId)->getAngleCount(); curAngle++) {
//            qreal curValue; QPoint curPos;
//            determineGlsWeightedMap(shuffledId, roughSolution, curAngle); QPoint newRoughPos = determineMinPos(shuffledId, curAngle, curValue);
//            getTotalZoomedOverlapMap(shuffledId, newRoughPos, curAngle, roughSolution);
//            curPos = determineMinZoomedPos(shuffledId, curAngle, curValue);
//            if(curValue < minValue) {
//                minValue = curValue; minPos = curPos; minAngle = curAngle;
//            }
//        }
//        solution.setOrientation(shuffledId, minAngle);
//        solution.setPosition(shuffledId, minPos);
//    }
//}
