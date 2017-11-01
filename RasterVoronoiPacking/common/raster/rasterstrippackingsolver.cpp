#include "rasterstrippackingsolver.h"
#include <QtCore/qmath.h>

using namespace RASTERVORONOIPACKING;

std::shared_ptr<RasterStripPackingSolver> RasterStripPackingSolver::createRasterPackingSolver(std::shared_ptr<RasterPackingProblem> problem, RasterStripPackingParameters &parameters, int initialWidth, int initialHeight) {
	std::shared_ptr<RasterStripPackingSolver> solver;
	std::shared_ptr<GlsWeightSet> weights;
	std::shared_ptr<RasterTotalOverlapMapEvaluator> overlapEvaluator;

	// Determine weight
	if (parameters.getHeuristic() == NONE) weights = std::shared_ptr<GlsNoWeightSet>(new GlsNoWeightSet); // No guided local search
	else weights = std::shared_ptr<GlsWeightSet>(new GlsWeightSet(problem->count())); // GLS

	// Determine overlap evaluator
	if (parameters.getZoomFactor() > 1 && parameters.getZoomFactor() <= problem->getScale())
		overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorDoubleGLS>(new RasterTotalOverlapMapEvaluatorDoubleGLS(problem, parameters.getZoomFactor(), weights, true));
	else overlapEvaluator = std::shared_ptr<RasterTotalOverlapMapEvaluatorGLS>(new RasterTotalOverlapMapEvaluatorGLS(problem, weights, true)); 
	
	// Create solver
	if (parameters.getClusterFactor() < 0)
		solver = std::shared_ptr<RasterStripPackingSolver>(new RasterStripPackingSolver(problem, overlapEvaluator));
	else {
		std::shared_ptr<RASTERVORONOIPACKING::RasterPackingClusterProblem> clusterProblem = std::dynamic_pointer_cast<RASTERVORONOIPACKING::RasterPackingClusterProblem>(problem);
		RasterStripPackingParameters nonClusterParameters; nonClusterParameters.Copy(parameters); nonClusterParameters.setClusterFactor(-1.0);
		std::shared_ptr<RasterStripPackingSolver> originalSolver = RasterStripPackingSolver::createRasterPackingSolver(clusterProblem->getOriginalProblem(), nonClusterParameters);
		solver = std::shared_ptr<RasterStripPackingSolver>(new RasterStripPackingClusterSolver(clusterProblem, overlapEvaluator, originalSolver));
	}

	if (initialWidth > 0 && initialHeight < 0) solver->setContainerWidth(initialWidth);
	else if (initialWidth > 0 && initialHeight > 0) solver->setContainerDimensions(initialWidth, initialHeight);
	return solver;
}

RasterStripPackingSolver::RasterStripPackingSolver(std::shared_ptr<RasterPackingProblem> _problem, std::shared_ptr<RasterTotalOverlapMapEvaluator> _overlapEvaluator) : overlapEvaluator(_overlapEvaluator) {
    this->originalProblem = _problem;
	currentWidth = this->originalProblem->getContainerWidth(); currentHeight = this->originalProblem->getContainerHeight();
	initialWidth = currentWidth; initialHeight = currentHeight;
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
		std::shared_ptr<RasterNoFitPolygon> ifp = originalProblem->getIfps()->getRasterNoFitPolygon(0, 0, originalProblem->getItemType(i), rnd_angle);
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
void RasterStripPackingSolver::generateBottomLeftSolution(RasterPackingSolution &solution, BottomLeftMode mode) {
	switch(mode) {
		case BL_STRIPPACKING: generateBottomLeftStripSolution(solution); break;
		case BL_RECTANGULAR: generateBottomLeftRectangleSolution(solution); break;
		case BL_SQUARE: generateBottomLeftSquareSolution(solution); break;
	}
}

void RasterStripPackingSolver::generateBottomLeftStripSolution(RasterPackingSolution &solution) {
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
			originalProblem->getIfpBoundingBox(shuffledId, angle, minIfpX, minIfpY, maxIfpX, maxIfpY);
			QPoint curPos(minIfpX, minIfpY);
			while (1) { // FIXME: Infinite loop?
				//if (qFuzzyCompare(1.0 + 0.0, 1.0 + overlap)) break;
				if (!detectItemPartialOverlap(sequence, k, curPos, angle, solution))
					break;
				// Get next position
				curPos.setY(curPos.y() + 1); if (curPos.y() > maxIfpY) { curPos.setY(minIfpY); curPos.setX(curPos.x() + 1); }
			}
			// Check minimum X coordinate
			int maxItemX = curPos.x() + qCeil(originalProblem->getScale()*originalProblem->getItem(shuffledId)->getMaxX(angle));
			if (angle == 0 || maxItemX < minMaxItemX) {
				minMaxItemX = maxItemX;
				solution.setPosition(shuffledId, curPos); solution.setOrientation(shuffledId, angle);
			}
		}
		if (minMaxItemX > layoutLength) layoutLength = minMaxItemX;
	}
	setContainerWidth(layoutLength, solution);
}

// Detect if item is in overlapping position for a subset of fixed items
bool RasterStripPackingSolver::detectItemPartialOverlap(QVector<int> sequence, int itemSequencePos, QPoint itemPos, int itemAngle, RasterPackingSolution &solution) {
	if (itemSequencePos == 0) return false;
	int itemId = sequence[itemSequencePos];
	for (int i = 0; i < itemSequencePos; i++) {
		int curFixedItemId = sequence[i];
		if (originalProblem->areOverlapping(itemId, itemPos, itemAngle, curFixedItemId, solution.getPosition(curFixedItemId), solution.getOrientation(curFixedItemId)))
			return true;
	}
	return false;
}

// --> Get layout overlap (sum of individual overlap values)
quint32 RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution) {
	quint32 totalOverlap = 0;
    for(int itemId = 0; itemId < originalProblem->count(); itemId++) {
		totalOverlap += getItemTotalOverlap(itemId, solution);
    }
    return totalOverlap;
}

quint32 RasterStripPackingSolver::getGlobalOverlap(RasterPackingSolution &solution, QVector<quint32> &overlaps, quint32 &maxOverlap) {
	maxOverlap = 0;
	quint32 totalOverlap = 0;
	int overlapIndex = 0;
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		quint32 curOverlap = 0;
		for (int i = 0; i < originalProblem->count(); i++) {
			if (i == itemId) { overlaps[overlapIndex++] = 0; continue; }
			quint32 individualOverlap = originalProblem->getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId), i, solution.getPosition(i), solution.getOrientation(i));
			overlaps[overlapIndex++] = individualOverlap;
			curOverlap += individualOverlap;
			if (individualOverlap > maxOverlap) maxOverlap = individualOverlap;
		}
		totalOverlap += curOverlap;
	}
	return totalOverlap;
}

bool RasterStripPackingSolver::setContainerWidth(int &pixelWidth, RasterPackingSolution &solution) {
	if (!setContainerWidth(pixelWidth)) return false;

	// Detect extruding items and move them horizontally back inside the container
	for (int itemId = 0; itemId < originalProblem->count(); itemId++) {
		std::shared_ptr<RasterNoFitPolygon> ifp = originalProblem->getIfps()->getRasterNoFitPolygon(0, 0, originalProblem->getItemType(itemId), solution.getOrientation(itemId));
		int maxPositionX = -ifp->getOriginX() + ifp->width() - (this->initialWidth - this->currentWidth) - 1;
		QPoint curItemPos = solution.getPosition(itemId);
		if (curItemPos.x() > maxPositionX) {
			curItemPos.setX(maxPositionX);
			solution.setPosition(itemId, curItemPos);
		}
	}

	return true;
}

bool RasterStripPackingSolver::setContainerWidth(int &pixelWidth) {
	// Check if size is smaller than smallest item width
	if (pixelWidth < this->getMinimumContainerWidth()) { pixelWidth = this->currentWidth; return false; }

	// Resize container
	overlapEvaluator->updateMapsLength(pixelWidth);
	this->currentWidth = pixelWidth;
	return true;
}

void RasterStripPackingSolver::performLocalSearch(RasterPackingSolution &solution) {
	QVector<int> sequence;
	for (int i = 0; i < originalProblem->count(); i++) sequence.append(i);
	std::random_shuffle(sequence.begin(), sequence.end());
	for (int i = 0; i < originalProblem->count(); i++) {
		int shuffledId = sequence[i];
		if (!detectItemTotalOverlap(shuffledId, solution)) continue;
		quint32 minValue; QPoint minPos; int minAngle = 0;
		minPos = getMinimumOverlapPosition(shuffledId, minAngle, solution, minValue);
		if (minValue != 0) {
			for (uint curAngle = 1; curAngle < originalProblem->getItem(shuffledId)->getAngleCount(); curAngle++) {
				quint32 curValue; QPoint curPos;
				curPos = getMinimumOverlapPosition(shuffledId, curAngle, solution, curValue);
				if (curValue < minValue) {
					minValue = curValue; minPos = curPos; minAngle = curAngle;
					if (minValue == 0) break;
				}
			}
		}
		solution.setOrientation(shuffledId, minAngle);
		solution.setPosition(shuffledId, minPos);
	}
}

// --> Get absolute minimum overlap position
QPoint RasterStripPackingSolver::getMinimumOverlapPosition(int itemId, int orientation, RasterPackingSolution &solution, quint32 &value) {
	std::shared_ptr<TotalOverlapMap> map = overlapEvaluator->getTotalOverlapMap(itemId, orientation, solution);
	QPoint minRelativePos;
	value = map->getMinimum(minRelativePos);
	return minRelativePos;
}

void getNextBLPosition(QPoint &curPos, int  minIfpX, int minIfpY, int maxIfpX, int maxIfpY) {
	curPos.setY(curPos.y() + 1); 
	if (curPos.y() > maxIfpY) { 
		curPos.setY(minIfpY); curPos.setX(curPos.x() + 1);
	}
}

quint32 RasterStripPackingSolver::getItemTotalOverlap(int itemId, RasterPackingSolution &solution) {
	quint32 totalOverlap = 0;
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		totalOverlap += originalProblem->getDistanceValue(itemId, solution.getPosition(itemId), solution.getOrientation(itemId),
			i, solution.getPosition(i), solution.getOrientation(i));
	}
	return totalOverlap;
}

bool RasterStripPackingSolver::detectItemTotalOverlap(int itemId, RasterPackingSolution &solution) {
	for (int i = 0; i < originalProblem->count(); i++) {
		if (i == itemId) continue;
		if (originalProblem->areOverlapping(itemId, solution.getPosition(itemId), solution.getOrientation(itemId), i, solution.getPosition(i), solution.getOrientation(i))) return true;
	}
	return false;
}