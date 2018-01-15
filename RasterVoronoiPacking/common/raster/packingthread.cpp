#include "packingthread.h"
#include <limits>
#include <QDebug>
#include <QTime>

#define MAXLOOPSPERLENGTH 5
#define UPDATEINTERVAL 0.2

PackingThread::~PackingThread() {
    m_abort = true;
    wait();
}
void PackingThread::setInitialSolution(RASTERVORONOIPACKING::RasterPackingSolution &initialSolution) {
    threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(initialSolution.getNumItems());
    for(int i = 0; i < initialSolution.getNumItems(); i++) {
        threadSolution.setPosition(i, initialSolution.getPosition(i));
        threadSolution.setOrientation(i, initialSolution.getOrientation(i));
    }
}

qreal PackingThread::getTimeStamp(int timeLimit, QDateTime &finalTime) {
	qreal ans = (timeLimit * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0;
	return ans;
}

void PackingThread::run()
{
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
    int itNum = 0;
    int totalItNum = 0;
    int worseSolutionsCount = 0;
	bool success = false;
	ExecutionSolutionInfo minSuccessfullSol(-1, -1, 0, seed, compactor->getProblemType());
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

	// Generate initial solution
	//if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) solver->generateRandomSolution(threadSolution);
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::BOTTOMLEFT)  {
		// Generate initial bottom left solution and determine the initial length
		compactor->generateBottomLeftSolution(threadSolution);
		minSuccessfullSol = ExecutionSolutionInfo(compactor->getCurrentLength(), compactor->getCurrentHeight(), getTimeStamp(parameters.getTimeLimit(), finalTime), 1, seed, compactor->getProblemType());
		emit minimumLenghtUpdated(threadSolution, minSuccessfullSol);
		bestSolution = threadSolution;

		// Execution the first container reduction
		compactor->shrinkContainer(threadSolution);
		emit solutionGenerated(threadSolution, ExecutionSolutionInfo(minSuccessfullSol.length, minSuccessfullSol.height, 1, seed, compactor->getProblemType()));
	}
	minOverlap = solver->getGlobalOverlap(threadSolution);
	itNum++; totalItNum++;

	qreal nextUpdateTime = QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 - UPDATEINTERVAL;
	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if(m_abort) break;
			solver->performLocalSearch(threadSolution);
			curOverlap = solver->getGlobalOverlap(threadSolution, currentOverlaps, maxItemOverlap);
			solver->updateWeights(threadSolution, currentOverlaps, maxItemOverlap);
			if (curOverlap < minOverlap || minOverlap == 0) {
				minOverlap = curOverlap;
				if (parameters.isFixedLength()) { minSuccessfullSol = ExecutionSolutionInfo(compactor->getCurrentLength(), getTimeStamp(parameters.getTimeLimit(), finalTime), totalItNum, seed, compactor->getProblemType()); bestSolution = threadSolution; }// Best solution for the minimum overlap problem
				if (curOverlap == 0) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			#ifndef CONSOLE
			if (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 < nextUpdateTime) {
				nextUpdateTime = nextUpdateTime - UPDATEINTERVAL;
				emit statusUpdated(compactor->getCurrentLength(), totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit()*1000-QDateTime::currentDateTime().msecsTo(finalTime))/1000.0);
				emit solutionGenerated(threadSolution, ExecutionSolutionInfo(compactor->getCurrentLength(), compactor->getCurrentHeight(), totalItNum, seed, compactor->getProblemType()));
				emit weightsChanged();
			}
			#endif
			itNum++; totalItNum++;
		}
		if (!m_abort && !parameters.isFixedLength()) {
			// Reduce or expand container
			if(success) {
				numLoops = 1;
				RASTERVORONOIPACKING::RasterPackingSolution tempBestSolution = threadSolution;
				int tempCurrentLength = compactor->getCurrentLength(); int tempCurrentHeight = compactor->getCurrentHeight();
				if (compactor->shrinkContainer(threadSolution)) {
					bestSolution = tempBestSolution;
					minSuccessfullSol = ExecutionSolutionInfo(tempCurrentLength, tempCurrentHeight, getTimeStamp(parameters.getTimeLimit(), finalTime), totalItNum, seed, compactor->getProblemType());
					emit minimumLenghtUpdated(bestSolution, minSuccessfullSol);
				}
			}
			else if (numLoops >= MAXLOOPSPERLENGTH) {
				numLoops = 1;
				if (!compactor->expandContainer(threadSolution)) compactor->generateRandomSolution(threadSolution); // Shuffle layout if length is maxed
			}
			else numLoops++;

			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution);
		}
		else {
			if (success) break;
			else {
				if (numLoops > MAXLOOPSPERLENGTH) { compactor->generateRandomSolution(threadSolution); numLoops = 1; }
				else numLoops++;
			}
		}
		itNum = 0; worseSolutionsCount = 0; solver->resetWeights();
	}
	if (m_abort) {qDebug() << "Aborted!"; quit();}
	else { emit finishedExecution(bestSolution, minSuccessfullSol, totalItNum, curOverlap, minOverlap, getTimeStamp(parameters.getTimeLimit(), finalTime)); quit(); }
}

void PackingThread::abort() {
	m_abort = true;
	wait();
}