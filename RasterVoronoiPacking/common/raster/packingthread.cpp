#include "packingthread.h"
#include <limits>
#include <QDebug>
#include <QTime>

#define MAXLOOPSPERLENGTH 5

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

void PackingThread::run()
{
	m_abort = false;
	seed = QDateTime::currentDateTime().toTime_t();
	qsrand(seed);
    int itNum = 0;
    int totalItNum = 0;
    int worseSolutionsCount = 0;
	bool success = false;
	int curLength = solver->getCurrentWidth();
	int minSuccessfullLength = curLength;
	qreal curRealLength = (qreal)minSuccessfullLength;
	qreal rdec = 0.04; qreal rinc = 0.01;
	qreal minOverlap = std::numeric_limits<qreal>::max();
	qreal curOverlap = minOverlap;
    RASTERVORONOIPACKING::RasterPackingSolution bestSolution = threadSolution;
    solver->resetWeights();
	int numLoops = 1;

	// Determine time to finish
	QDateTime finalTime = QDateTime::currentDateTime();
	finalTime = finalTime.addSecs(parameters.getTimeLimit());

	// Generate initial solution
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::RANDOMFIXED) solver->generateRandomSolution(threadSolution, parameters);
	if (parameters.getInitialSolMethod() == RASTERVORONOIPACKING::BOTTOMLEFT)  {
		// Generate initial bottom left solution and determine the initial length
		solver->generateBottomLeftSolution(threadSolution, parameters);
		curLength = solver->getCurrentWidth();
		minSuccessfullLength = curLength;
		emit minimumLenghtUpdated(threadSolution, minSuccessfullLength, 1, 0, seed);
		// Execution the first container reduction
		curRealLength = (1.0 - rdec)*(qreal)solver->getCurrentWidth();
		curLength = qRound(curRealLength);
		solver->setContainerWidth(curLength, threadSolution, parameters);
	}
	minOverlap = solver->getGlobalOverlap(threadSolution, parameters);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, curLength);

	while (QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
		if (m_abort) break;
		while (worseSolutionsCount < parameters.getNmo() && QDateTime::currentDateTime().msecsTo(finalTime) / 1000.0 > 0 && (parameters.getIterationsLimit() == 0 || totalItNum < parameters.getIterationsLimit()) && !m_abort) {
			if(m_abort) break;
			solver->performLocalSearch(threadSolution, parameters);
			if (parameters.getHeuristic() == RASTERVORONOIPACKING::GLS)  solver->updateWeights(threadSolution, parameters);
			curOverlap = solver->getGlobalOverlap(threadSolution, parameters);
			if(curOverlap < minOverlap) {
				minOverlap = curOverlap;
				if(parameters.isFixedLength()) bestSolution = threadSolution; // Best solution for the minimum overlap problem
				if (qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) { success = true; break; }
				worseSolutionsCount = 0;
			}
			else worseSolutionsCount++;
			if(itNum % 10 == 0) {
				emit statusUpdated(curLength, totalItNum, worseSolutionsCount, curOverlap, minOverlap, (parameters.getTimeLimit()*1000-QDateTime::currentDateTime().msecsTo(finalTime))/1000.0);
				emit solutionGenerated(threadSolution, curLength);
				emit weightsChanged();
			}
			itNum++; totalItNum++;
		}
		if (!parameters.isFixedLength()) {
			// Reduce or expand container
			if(success) {
				numLoops = 1;
				bestSolution = threadSolution; minSuccessfullLength = curLength;
				curRealLength = (1.0 - rdec)*(qreal)solver->getCurrentWidth();
				curLength = qRound(curRealLength);
				emit minimumLenghtUpdated(bestSolution, minSuccessfullLength, totalItNum, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
			}
			else if (numLoops >= MAXLOOPSPERLENGTH) {
				numLoops = 1;
				if (qRound((1.0 + rinc)*curRealLength) >= minSuccessfullLength) {
					curRealLength = (curRealLength + minSuccessfullLength) / 2;
					if (qRound(curRealLength) == minSuccessfullLength)
						curRealLength = minSuccessfullLength - 1;
					if (curLength == qRound(curRealLength)) solver->generateRandomSolution(threadSolution, parameters);
					curLength = qRound(curRealLength);
				}
				else {
					curRealLength = (1.0 + rinc)*curRealLength;
					curLength = qRound(curRealLength);
				}
			}
			else numLoops++;

			solver->setContainerWidth(curLength, threadSolution, parameters);
			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution, parameters);
		}
		else {
			if (success) break;
			else {
				if (numLoops > MAXLOOPSPERLENGTH) { solver->generateRandomSolution(threadSolution, parameters); numLoops = 1; }
				else numLoops++;
			}
		}
		itNum = 0; worseSolutionsCount = 0; solver->resetWeights();
		emit weightsChanged();
	}
	if (m_abort) {qDebug() << "Aborted!"; quit();}
	solver->setContainerWidth(minSuccessfullLength, bestSolution, parameters);
	emit finishedExecution(bestSolution, minSuccessfullLength, totalItNum, curOverlap, minOverlap, (parameters.getTimeLimit() * 1000 - QDateTime::currentDateTime().msecsTo(finalTime)) / 1000.0, seed);
	quit();
}

void PackingThread::abort() {
	m_abort = true;
	wait();
}