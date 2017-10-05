#include "packing2dthread.h"
#include <limits>
#include <QDebug>
#include <QTime>

#define MAXLOOPSPERLENGTH 5
#define UPDATEINTERVAL 0.5

bool Packing2DThread::getShrinkedDimension(int dim, int &newDim, bool length) {
	// Get minimum dimension
	int minimumDimension = length ? solver2d->getMinimumContainerWidth() : solver2d->getMinimumContainerHeight();
	
	// Check if dimension is already minimum
	if (dim == minimumDimension) return false;
	
	// Get next dimension and check limit
	newDim = newDim < minimumDimension ? minimumDimension : newDim;
	return true;
}

void Packing2DThread::randomChangeContainerDimensions(int &curLenght, int &curHeight, const qreal ratio) {
	bool changeLenght = qrand() % 2 - 1;

	if (ratio > 1) {
		if(changeLenght) curLenght = std::ceil(ratio * (qreal)curLenght);
		else curHeight = std::ceil(ratio * (qreal)curHeight);
	}
	else{
		int reducedLength, reducedHeight;
		if (!checkShrinkSizeConstraint(curLenght, curHeight, reducedLength, reducedHeight, ratio)) return;
		if (changeLenght) curLenght = reducedLength;
		else curHeight = reducedHeight;
	}
}

void expandSmallerDimension(int &curLenght, int &curHeight, const qreal ratio) {
	curLenght = std::ceil(sqrt(ratio) * (qreal)curLenght);
	curHeight = std::ceil(sqrt(ratio) * (qreal)curHeight);
}

bool Packing2DThread::checkShrinkSizeConstraint(int &curLength, int &curHeight, int &reducedLength, int &reducedHeight, qreal ratio) {
	// Verify size constraints
	reducedLength = std::floor(ratio * (qreal)curLength);
	reducedHeight = std::floor(ratio * (qreal)curHeight);
	bool isPossibleToReduceLength = getShrinkedDimension(curLength, reducedLength, true);
	bool isPossibleToReduceHeight = getShrinkedDimension(curHeight, reducedHeight, false);
	if (!isPossibleToReduceLength && isPossibleToReduceHeight) curHeight = reducedHeight;
	if (!isPossibleToReduceHeight && isPossibleToReduceLength) curLength = reducedLength;
	if (!isPossibleToReduceLength || !isPossibleToReduceHeight) return false;
	return true;
}

void Packing2DThread::costShrinkContainerDimensions(int &curLenght, int &curHeight, RASTERVORONOIPACKING::RasterPackingSolution &currentSolution, const qreal ratio) {
	RASTERVORONOIPACKING::RasterPackingSolution tempSolution = currentSolution;

	// Verify size constraints
	int reducedLength, reducedHeight;
	if (!checkShrinkSizeConstraint(curLenght, curHeight, reducedLength, reducedHeight, ratio)) return;
	// FIXME: What if both are not reduceable??

	// Estimate cost in X
	this->solver2d->setContainerDimensions(reducedLength, curHeight, tempSolution, parameters);
	qreal costX = this->solver2d->getGlobalOverlap(tempSolution, parameters);

	// Reset container and solution
	this->solver2d->setContainerDimensions(curLenght, curHeight, tempSolution, parameters);
	tempSolution = currentSolution;

	// Estimate cost in Y
	this->solver2d->setContainerDimensions(curLenght, reducedHeight, tempSolution, parameters);
	qreal costY = this->solver2d->getGlobalOverlap(tempSolution, parameters);

	// Determine new height and length
	if (costX > costY) curHeight = reducedHeight;
	else curLenght = reducedLength;
}

void Packing2DThread::run() {

	switch (method) {
	case RASTERVORONOIPACKING::SQUARE:
		runSquare();
		break;
	case RASTERVORONOIPACKING::RANDOM_ENCLOSED: 
		runRandom();
		break;
	case RASTERVORONOIPACKING::BAGPIPE:
		runBagpipe();
		break;
	case RASTERVORONOIPACKING::COST_EVALUATION:
		runCost();
		break;
	}
	quit();
}

void Packing2DThread::runSquare() {
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
	int itNum = 0;
	int totalItNum = 0;
	int worseSolutionsCount = 0;
	bool success = false;
	int curDim;
	qreal rdec = 0.10; qreal rinc = 0.01;
	qreal areaDec = sqrt(1 - rdec), areaInc = sqrt(1 + rinc);
	qreal minOverlap = std::numeric_limits<qreal>::max();
	qreal curOverlap = minOverlap;
	QPair<RASTERVORONOIPACKING::RasterPackingSolution, int> lastFeasibleSolution, bestSolution;
	solver2d->resetWeights();
	int numLoops = 1;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial bottom left solution and determine the initial length
	solver2d->generateBottomLeftSquareSolution(threadSolution, parameters);
	curDim = solver2d->getCurrentWidth();
	lastFeasibleSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curDim);
	bestSolution = lastFeasibleSolution;
	// Create solution snapshot
	Solution2DInfo solInfo; solInfo.iteration = 1; solInfo.timestamp = 0;
	solInfo.length = curDim; solInfo.height = curDim; solInfo.area = solInfo.length * solInfo.height;
	emit dimensionUpdated(threadSolution, solInfo, 1, 0, seed);
	// Execution the first container reduction
	curDim = qRound(areaDec * solver2d->getCurrentWidth());
	solver2d->setContainerDimensions(curDim, curDim, threadSolution, parameters);

	minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, curDim, curDim);

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if (m_abort) break;
			solver2d->performLocalSearch(threadSolution, parameters);
			solver2d->updateWeights(threadSolution, parameters);
			curOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
			if (curOverlap < minOverlap || qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) {
				minOverlap = curOverlap;
				if (qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			//if(itNum % 10 == 0) {
			if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
				nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
				emit statusUpdated(curDim, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0);
				emit solutionGenerated(threadSolution, curDim, curDim);
				emit weightsChanged();
			}
			itNum++; totalItNum++;
		}
		// Create solution snapshot
		solInfo.iteration = totalItNum;
		solInfo.timestamp = (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0;
		if (!parameters.isFixedLength()) {
			// Reduce or expand container
			if (success) {
				solInfo.length = curDim; solInfo.height = curDim; solInfo.area = solInfo.length * solInfo.height;
				emit dimensionUpdated(threadSolution, solInfo, totalItNum, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
				numLoops = 1;
				if (solver2d->getCurrentWidth() < bestSolution.second)
					bestSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, solver2d->getCurrentWidth());
				curDim = qRound(areaDec * solver2d->getCurrentWidth());
				lastFeasibleSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curDim);
			}
			else if (numLoops >= MAXLOOPSPERLENGTH) {
				numLoops = 1;
				curDim = std::ceil(areaInc * solver2d->getCurrentWidth());
				// Failure
				if (lastFeasibleSolution.second > curDim) {
					solver2d->setContainerDimensions(lastFeasibleSolution.second, lastFeasibleSolution.second, threadSolution, parameters);
					threadSolution = lastFeasibleSolution.first;
				}
			}
			else numLoops++;

			solver2d->setContainerDimensions(curDim, curDim, threadSolution, parameters);
			success = false;
			minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
		}
		itNum = 0; worseSolutionsCount = 0; solver2d->resetWeights();
		emit weightsChanged();
	}
	if (m_abort) { qDebug() << "Aborted!"; quit(); }
	solver2d->setContainerWidth(bestSolution.second, bestSolution.first, parameters);
	emit finishedExecution(bestSolution.first, bestSolution.second, totalItNum, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
	quit();
}

void Packing2DThread::runRandom() {
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
	int itNum = 0;
	int totalItNum = 0;
	int worseSolutionsCount = 0;
	bool success = false;
	int curLenght, curHeight;
	qreal rdec = 0.10; qreal rinc = 0.01;
	const qreal areaDec = 1 - rdec, areaInc = 1 + rinc;
	qreal minOverlap = std::numeric_limits<qreal>::max();
	qreal curOverlap = minOverlap;
	//QPair<RASTERVORONOIPACKING::RasterPackingSolution, int> lastFeasibleSolution, bestSolution;
	QPair<RASTERVORONOIPACKING::RasterPackingSolution, int> bestSolution;
	solver2d->resetWeights();
	int numLoops = 1;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial bottom left solution and determine the initial length
	solver2d->generateBottomLeftSolution(threadSolution, parameters);
	curLenght = solver2d->getCurrentWidth(); curHeight = solver2d->getCurrentHeight();
	//lastFeasibleSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curLenght * curHeight);
	//bestSolution = lastFeasibleSolution;
	bestSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curLenght * curHeight);
	// Create solution snapshot
	Solution2DInfo solInfo; solInfo.iteration = 1; solInfo.timestamp = 0;
	solInfo.length = curLenght; solInfo.height = curHeight; solInfo.area = solInfo.length * solInfo.height;
	emit dimensionUpdated(threadSolution, solInfo, 1, 0, seed);
	// Execution the first container reduction
	randomChangeContainerDimensions(curLenght, curHeight, areaDec);
	solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);

	minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, curLenght, curHeight);

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if (m_abort) break;
			solver2d->performLocalSearch(threadSolution, parameters);
			solver2d->updateWeights(threadSolution, parameters);
			curOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
			if (curOverlap < minOverlap || qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) {
				minOverlap = curOverlap;
				if (qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			//if(itNum % 10 == 0) {
			if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
				nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
				emit statusUpdated(curLenght, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0);
				emit solutionGenerated(threadSolution, curLenght, curHeight);
				emit weightsChanged();
			}
			itNum++; totalItNum++;
		}

		// Create solution snapshot
		solInfo.iteration = totalItNum;
		solInfo.timestamp = (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0;
		// Reduce or expand container
		int currentArea = solver2d->getCurrentWidth() * solver2d->getCurrentHeight();
		if (success) {
			solInfo.length = curLenght; solInfo.height = curHeight; solInfo.area = solInfo.length * solInfo.height;
			emit dimensionUpdated(threadSolution, solInfo, totalItNum, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
			numLoops = 1;
			if (currentArea < bestSolution.second) bestSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, currentArea);
			randomChangeContainerDimensions(curLenght, curHeight, areaDec);
		}
		else if (numLoops >= MAXLOOPSPERLENGTH) {
			numLoops = 1;
			randomChangeContainerDimensions(curLenght, curHeight, areaInc);
		}
		else numLoops++;

		solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);
		success = false;
		minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);

		itNum = 0; worseSolutionsCount = 0; solver2d->resetWeights();
		emit weightsChanged();
	}
	if (m_abort) { qDebug() << "Aborted!"; quit(); }
	solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);
	emit finishedExecution(bestSolution.first, bestSolution.second, totalItNum, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
}

void Packing2DThread::runCost() {
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
	int itNum = 0;
	int totalItNum = 0;
	int worseSolutionsCount = 0;
	bool success = false;
	int curLenght, curHeight;
	qreal rdec = 0.10; qreal rinc = 0.01;
	const qreal areaDec = 1 - rdec, areaInc = 1 + rinc;
	qreal minOverlap = std::numeric_limits<qreal>::max();
	qreal curOverlap = minOverlap;
	QPair<RASTERVORONOIPACKING::RasterPackingSolution, int> bestSolution;
	solver2d->resetWeights();
	int numLoops = 1;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial bottom left solution and determine the initial length
	solver2d->generateBottomLeftSolution(threadSolution, parameters);
	curLenght = solver2d->getCurrentWidth(); curHeight = solver2d->getCurrentHeight();
	//lastFeasibleSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curLenght * curHeight);
	//bestSolution = lastFeasibleSolution;
	bestSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, curLenght * curHeight);
	// Create solution snapshot
	Solution2DInfo solInfo; solInfo.iteration = 1; solInfo.timestamp = 0;
	solInfo.length = curLenght; solInfo.height = curHeight; solInfo.area = solInfo.length * solInfo.height;
	emit dimensionUpdated(threadSolution, solInfo, 1, 0, seed);
	// Execution the first container reduction
	costShrinkContainerDimensions(curLenght, curHeight, threadSolution, areaDec);
	solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);

	minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, curLenght, curHeight);

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if (m_abort) break;
			solver2d->performLocalSearch(threadSolution, parameters);
			solver2d->updateWeights(threadSolution, parameters);
			curOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);
			if (curOverlap < minOverlap || qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) {
				minOverlap = curOverlap;
				if (qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			//if(itNum % 10 == 0) {
			if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
				nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
				emit statusUpdated(curLenght, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0);
				emit solutionGenerated(threadSolution, curLenght, curHeight);
				emit weightsChanged();
			}
			itNum++; totalItNum++;
		}

		// Create solution snapshot
		solInfo.iteration = totalItNum;
		solInfo.timestamp = (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0;
		// Reduce or expand container
		int currentArea = solver2d->getCurrentWidth() * solver2d->getCurrentHeight();
		if (success) {
			solInfo.length = curLenght; solInfo.height = curHeight; solInfo.area = solInfo.length * solInfo.height;
			emit dimensionUpdated(threadSolution, solInfo, totalItNum, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
			numLoops = 1;
			if (currentArea < bestSolution.second) bestSolution = QPair<RASTERVORONOIPACKING::RasterPackingSolution, int>(threadSolution, currentArea);
			costShrinkContainerDimensions(curLenght, curHeight, threadSolution, areaDec);
		}
		else if (numLoops >= MAXLOOPSPERLENGTH) {
			numLoops = 1;
			expandSmallerDimension(curLenght, curHeight, areaInc);
		}
		else numLoops++;

		solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);
		success = false;
		minOverlap = solver2d->getGlobalOverlap(threadSolution, parameters);

		itNum = 0; worseSolutionsCount = 0; solver2d->resetWeights();
		emit weightsChanged();
	}
	if (m_abort) { qDebug() << "Aborted!"; quit(); }
	solver2d->setContainerDimensions(curLenght, curHeight, threadSolution, parameters);
	emit finishedExecution(bestSolution.first, bestSolution.second, totalItNum, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
}

void Packing2DThread::runBagpipe() {
}