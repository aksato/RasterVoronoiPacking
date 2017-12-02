#include "packing2dthread.h"
#include <cmath>
#include <limits>
#include <QDebug>
#include <QTime>
#include <QtMath>

#define MAXLOOPSPERLENGTH 5
#define UPDATEINTERVAL 0.2

bool getShrinkedDimension(qreal realDim, qreal &newRealDim, int minimumDimension) {
	// Check if dimension is already minimum
	if (qRound(realDim) == minimumDimension) return false;

	// Get next dimension and check limit
	if (qRound(newRealDim) < minimumDimension) newRealDim = minimumDimension;

	return true;
}

void Packing2DThread::randomChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, const qreal ratio) {
	bool changeLength = qrand() % 2 - 1;

	// Expansion
	if (ratio > 1) {
		if (changeLength) { curRealLength = ratio * curRealLength; curLength = qRound(curRealLength); }
		else { curRealHeight = ratio * curRealHeight; curHeight = qRound(curRealHeight); }
		return;
	}

	// Reduction
	if (changeLength) {
		qreal reducedRealLength = ratio * curRealLength;
		if (getShrinkedDimension(curRealLength, reducedRealLength, solver->getMinimumContainerWidth())) {
			curRealLength = reducedRealLength; curLength = qRound(curRealLength);
			return;
		}
	}
	qreal reducedRealHeight = ratio * curRealHeight;
	if (getShrinkedDimension(curRealHeight, reducedRealHeight, solver->getMinimumContainerHeight())) {
		curRealHeight = reducedRealHeight; curHeight = qRound(curRealHeight);
		return;
	}
	qreal reducedRealLength = ratio * curRealLength;
	if (getShrinkedDimension(curRealLength, reducedRealLength, solver->getMinimumContainerWidth())) {
		curRealLength = reducedRealLength; curLength = qRound(curRealLength);
	}
}

void Packing2DThread::costChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, RASTERVORONOIPACKING::RasterPackingSolution &currentSolution, const qreal ratio) {
	// Expansion
	if (ratio > 1) {
		// Keep aspect ratio
		curRealLength = sqrt(ratio) * curRealLength; curLength = qRound(curRealLength);
		curRealHeight = sqrt(ratio) * curRealHeight; curHeight = qRound(curRealHeight);
		return;
	}

	// Reduction
	RASTERVORONOIPACKING::RasterPackingSolution tempSolution = currentSolution;
	qreal costX, costY;
	qreal reducedRealLength = ratio * curRealLength;
	qreal reducedRealHeight = ratio * curRealHeight;

	// Estimate cost in X
	if (getShrinkedDimension(curRealLength, reducedRealLength, solver->getMinimumContainerWidth())) {
		int reducedLength = qRound(reducedRealLength);
		this->solver->setContainerDimensions(reducedLength, curHeight, tempSolution);
		costX = this->solver->getGlobalOverlap(tempSolution);
	}
	else {
		// FIXME: What if both dimensions are minimum!
		curRealHeight = reducedRealHeight; curHeight = qRound(curRealHeight);
		return;
	}

	// Reset container and solution
	this->solver->setContainerDimensions(curLength, curHeight, tempSolution);
	tempSolution = currentSolution;

	// Estimate cost in Y
	if (getShrinkedDimension(curRealHeight, reducedRealHeight, solver->getMinimumContainerHeight())) {
		int reducedHeight = qRound(reducedRealHeight);
		this->solver->setContainerDimensions(curLength, reducedHeight, tempSolution);
		costY = this->solver->getGlobalOverlap(tempSolution);
	}
	else return; // FIXME: Should never happen with morethan one item!

	// Determine new height and length
	if (costX > costY) { curRealHeight = reducedRealHeight; curHeight = qRound(curRealHeight); }
	else { curRealLength = reducedRealLength; curLength = qRound(curRealLength); }
}

void Packing2DThread::bagpipeChangeContainerDimensions(int &curLength, int &curHeight, qreal &curRealLength, qreal &curRealHeight, int bestArea, const qreal ratio) {
	if (ratio > 1) {
		// Keep aspect ratio
		qreal expandedRealLength = sqrt(ratio) * curRealLength;
		qreal expandedRealHeight = sqrt(ratio) * curRealHeight;
		if (qRound(expandedRealLength) * qRound(expandedRealHeight) < bestArea) {
			curRealHeight = expandedRealHeight; curHeight = qRound(curRealHeight);
			curRealLength = expandedRealLength; curLength = qRound(curRealLength);
			return;
		}

		qreal rinc = ratio - 1.0;
		qreal expansionDelta = qMax(rinc*curRealLength, rinc*curRealHeight);
		qreal curArea = curRealLength * curRealHeight;
		if (bagpipeDirection) {
			expandedRealHeight = curRealHeight + expansionDelta;
			qreal reducedRealLength = curArea / expandedRealHeight;
			getShrinkedDimension(curRealLength, reducedRealLength, solver->getMinimumContainerWidth());

			curRealLength = reducedRealLength; curLength = qRound(curRealLength);
			curRealHeight = expandedRealHeight;  curHeight = qRound(curRealHeight);
		}
		else {
			expandedRealLength = curRealLength + expansionDelta;
			qreal reducedRealHeight = curArea / expandedRealLength;
			getShrinkedDimension(curRealHeight, reducedRealHeight, solver->getMinimumContainerHeight());

			curRealLength = expandedRealLength; curLength = qRound(curRealLength);
			curRealHeight = reducedRealHeight; curHeight = qRound(curRealHeight);
		}
		if (curLength == solver->getMinimumContainerWidth() || curHeight == solver->getMinimumContainerHeight()) {
			bagpipeDirection = !bagpipeDirection;
		}
		return;
	}

	// Reduction
	qreal reducedRealLength = ratio * curRealLength;
	qreal reducedRealHeight = ratio * curRealHeight;
	if (getShrinkedDimension(curRealLength, reducedRealLength, solver->getMinimumContainerWidth()) && getShrinkedDimension(curRealHeight, reducedRealHeight, solver->getMinimumContainerHeight())) {
		curRealHeight = reducedRealHeight; curHeight = qRound(curRealHeight);
		curRealLength = reducedRealLength; curLength = qRound(curRealLength);
	}
}

void Packing2DThread::run() {

	switch (parameters.getRectangularPackingMethod()) {
	case RASTERVORONOIPACKING::SQUARE:
		runSquare();
		break;
	case RASTERVORONOIPACKING::RANDOM_ENCLOSED: case RASTERVORONOIPACKING::COST_EVALUATION: case RASTERVORONOIPACKING::BAGPIPE:
		runRectangle();
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
	int curDim = solver->getCurrentWidth();
	ExecutionSolutionInfo minSuccessfullSol(curDim, curDim, 0, seed);
	qreal curRealDim = (qreal)minSuccessfullSol.length;
	qreal rdec = parameters.getRdec(); qreal rinc = parameters.getRinc();
	qreal areaDec = sqrt(1 - rdec), areaInc = sqrt(1 + rinc);
	qreal minOverlap = std::numeric_limits<qreal>::max();
	qreal curOverlap = minOverlap;
	RASTERVORONOIPACKING::RasterPackingSolution bestSolution = threadSolution;
	solver->resetWeights();
	int numLoops = 1;
	QVector<quint32> currentOverlaps(solver->getNumItems()*solver->getNumItems());
	quint32 maxItemOverlap;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial bottom left solution and determine the initial length
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) solver->generateRandomSolution(threadSolution);
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::BOTTOMLEFT)  {
		solver->generateBottomLeftSolution(threadSolution, RASTERVORONOIPACKING::BL_SQUARE);
		curDim = solver->getCurrentWidth();
		minSuccessfullSol = ExecutionSolutionInfo(curDim, curDim, getTimeStamp(parameters.getTimeLimit(), finalTime), 1, seed);
		emit minimumLenghtUpdated(threadSolution, minSuccessfullSol);
		bestSolution = threadSolution;

		// Execution the first container reduction
		curRealDim = areaDec*(qreal)solver->getCurrentWidth();
		curDim = qRound(curRealDim);
		solver->setContainerDimensions(curDim, curDim, threadSolution);
	}
	minOverlap = solver->getGlobalOverlap(threadSolution);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, ExecutionSolutionInfo(curDim, curDim, 1, seed));

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if(m_abort) break;
			solver->performLocalSearch(threadSolution);
			curOverlap = solver->getGlobalOverlap(threadSolution, currentOverlaps, maxItemOverlap);
			solver->updateWeights(threadSolution, currentOverlaps, maxItemOverlap);
			if(curOverlap < minOverlap) {
				minOverlap = curOverlap;
				if(parameters.isFixedLength()) bestSolution = threadSolution; // Best solution for the minimum overlap problem
				if (curOverlap == 0) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			#ifndef CONSOLE
			if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
				nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
				emit statusUpdated(curDim, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit()*1000-QDateTime::currentDateTime().msecsTo(finalTime))/1000.0);
				emit solutionGenerated(threadSolution, ExecutionSolutionInfo(curDim, curDim, totalItNum, seed));
				emit weightsChanged();
			}
			#endif
			itNum++; totalItNum++;
		}
		// Create solution snapshot
		if (!parameters.isFixedLength()) {
			// Reduce or expand container
			if (success) {
				numLoops = 1;
				bestSolution = threadSolution; minSuccessfullSol = ExecutionSolutionInfo(curDim, curDim, getTimeStamp(parameters.getTimeLimit(), finalTime), totalItNum, seed);
				curRealDim = areaDec*(qreal)solver->getCurrentWidth();
				curDim = qRound(curRealDim);
				emit minimumLenghtUpdated(bestSolution, minSuccessfullSol);
			}
			else if (numLoops >= MAXLOOPSPERLENGTH) {
				numLoops = 1;
				if (qRound(areaInc*curRealDim) >= minSuccessfullSol.length) {
					curRealDim = (curRealDim + minSuccessfullSol.length) / 2;
					if (qRound(curRealDim) == minSuccessfullSol.length)
						curRealDim = minSuccessfullSol.length - 1;
					if (curDim == qRound(curRealDim)) solver->generateRandomSolution(threadSolution);
					curDim = qRound(curRealDim);
				}
				else {
					curRealDim = areaInc*curRealDim;
					curDim = qRound(curRealDim);
				}
			}
			else numLoops++;

			solver->setContainerDimensions(curDim, curDim, threadSolution);
			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution);
		}
		else {
			if (success) break;
			else {
				if (numLoops > MAXLOOPSPERLENGTH) { solver->generateRandomSolution(threadSolution); numLoops = 1; }
				else numLoops++;
			}
		}
		itNum = 0; worseSolutionsCount = 0; solver->resetWeights();
		//emit weightsChanged();
	}
	if (m_abort) { qDebug() << "Aborted!"; quit(); }
	else {
		solver->setContainerDimensions(minSuccessfullSol.length, minSuccessfullSol.length,  bestSolution);
		emit finishedExecution(bestSolution, minSuccessfullSol, totalItNum, curOverlap, minOverlap, getTimeStamp(parameters.getTimeLimit(), finalTime));
		quit();
	}
}

void Packing2DThread::runRectangle() {
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
	int itNum = 0;
	int totalItNum = 0;
	int worseSolutionsCount = 0;
	bool success = false;
	int curLength = solver->getCurrentWidth();
	int curHeight = solver->getCurrentHeight();
	ExecutionSolutionInfo minSuccessfullSol(curLength, curHeight, 0, seed);
	qreal curRealLength = (qreal)minSuccessfullSol.length;
	qreal curRealHeight = (qreal)minSuccessfullSol.height;
	qreal rdec = parameters.getRdec(); qreal rinc = parameters.getRinc();
	const qreal areaDec = 1 - rdec, areaInc = 1 + rinc;
	quint32 minOverlap = std::numeric_limits<quint32>::max();
	quint32 curOverlap = minOverlap;
	RASTERVORONOIPACKING::RasterPackingSolution bestSolution = threadSolution;
	solver->resetWeights();
	int numLoops = 1;
	QVector<quint32> currentOverlaps(solver->getNumItems()*solver->getNumItems());
	quint32 maxItemOverlap;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial bottom left solution and determine the initial length
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) solver->generateRandomSolution(threadSolution);
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::BOTTOMLEFT)  {
		solver->generateBottomLeftSolution(threadSolution, RASTERVORONOIPACKING::BL_RECTANGULAR);
		curRealLength = (qreal)solver->getCurrentWidth(); curRealHeight = (qreal)solver->getCurrentHeight();
		curLength = solver->getCurrentWidth(); curHeight = solver->getCurrentHeight();
		minSuccessfullSol = ExecutionSolutionInfo(curLength, curHeight, getTimeStamp(parameters.getTimeLimit(), finalTime), 1, seed);
		emit minimumLenghtUpdated(threadSolution, minSuccessfullSol);
		bestSolution = threadSolution;

		// Execution the first container reduction
		switch (parameters.getRectangularPackingMethod()) {
			case RASTERVORONOIPACKING::RANDOM_ENCLOSED: randomChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, areaDec); break;
			case RASTERVORONOIPACKING::COST_EVALUATION: costChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, threadSolution, areaDec); break;
			case RASTERVORONOIPACKING::BAGPIPE: bagpipeChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, minSuccessfullSol.area, areaDec); break;
		}
		solver->setContainerDimensions(curLength, curHeight, threadSolution);
	}
	minOverlap = solver->getGlobalOverlap(threadSolution);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, ExecutionSolutionInfo(curLength, curHeight, 1, seed));

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		if (minOverlap != 0) {
			while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
				if (m_abort) break;
				solver->performLocalSearch(threadSolution);
				curOverlap = solver->getGlobalOverlap(threadSolution, currentOverlaps, maxItemOverlap);
				solver->updateWeights(threadSolution, currentOverlaps, maxItemOverlap);
				if (curOverlap < minOverlap) {
					minOverlap = curOverlap;
					if (parameters.isFixedLength()) bestSolution = threadSolution; // Best solution for the minimum overlap problem
					if (curOverlap == 0) { success = true; break; }
					worseSolutionsCount = 0;
				}
				else worseSolutionsCount++;
				#ifndef CONSOLE
				if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
					nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
					emit statusUpdated(curLength, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0);
					emit solutionGenerated(threadSolution, ExecutionSolutionInfo(curLength, curHeight, totalItNum, seed));
					emit weightsChanged();
				}
				#endif
				itNum++; totalItNum++;
			}
		}
		else success = true;
		if (!parameters.isFixedLength()) {
			// Reduce or expand container
			int currentArea = solver->getCurrentWidth() * solver->getCurrentHeight();
			qreal ratio;
			if (success) {
				numLoops = 1;
				if (currentArea < minSuccessfullSol.area) {
					bestSolution = threadSolution; minSuccessfullSol = ExecutionSolutionInfo(curLength, curHeight, getTimeStamp(parameters.getTimeLimit(), finalTime), totalItNum, seed);
					emit minimumLenghtUpdated(bestSolution, minSuccessfullSol);
				}
				ratio = areaDec;
			}
			else if (numLoops >= MAXLOOPSPERLENGTH) {
				numLoops = 1;
				ratio = areaInc;
			}
			else numLoops++;
			if (numLoops == 1) {
				switch (parameters.getRectangularPackingMethod()) {
					case RASTERVORONOIPACKING::RANDOM_ENCLOSED: randomChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, ratio); break;
					case RASTERVORONOIPACKING::COST_EVALUATION: costChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, threadSolution, ratio); break;
					case RASTERVORONOIPACKING::BAGPIPE: bagpipeChangeContainerDimensions(curLength, curHeight, curRealLength, curRealHeight, minSuccessfullSol.area, ratio); break;
				}
			}

			solver->setContainerDimensions(curLength, curHeight, threadSolution);
			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution);
		}
		else {
			if (success) break;
			else {
				if (numLoops > MAXLOOPSPERLENGTH) { solver->generateRandomSolution(threadSolution); numLoops = 1; }
				else numLoops++;
			}
		}
		itNum = 0; worseSolutionsCount = 0; solver->resetWeights();
		//emit weightsChanged();
	}
	if (m_abort) { qDebug() << "Aborted!"; quit(); }
	else {
		solver->setContainerDimensions(minSuccessfullSol.length, minSuccessfullSol.height, bestSolution);
		emit finishedExecution(bestSolution, minSuccessfullSol, totalItNum, curOverlap, minOverlap, getTimeStamp(parameters.getTimeLimit(), finalTime));
		quit();
	}
}
