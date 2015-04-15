#include "packingthread.h"
#include <QDebug>
#include <QTime>

#define MAXLOOPSPERLENGTH 5

#ifndef CONSOLE
PackingThread::PackingThread(QObject *parent) :
    QThread(parent)
#else
#include <iostream>
PackingThread::PackingThread(QObject *parent) :
    QObject(parent)
#endif
{
    finishNow = false;
    seed = QDateTime::currentDateTime().toTime_t();// seed = 561; //TOREMOVE
    qDebug() << "Seed:" << seed;
    qsrand(seed);
}

PackingThread::~PackingThread() {
    finishNow = true;
    #ifndef CONSOLE
        wait();
    #endif
}
void PackingThread::setInitialSolution(RASTERVORONOIPACKING::RasterPackingSolution &initialSolution) {
    threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(initialSolution.getNumItems());
    for(int i = 0; i < initialSolution.getNumItems(); i++) {
        threadSolution.setPosition(i, initialSolution.getPosition(i));
        threadSolution.setOrientation(i, initialSolution.getOrientation(i));
    }
}

#ifndef CONSOLE
void PackingThread::run()
#else
void PackingThread::run(qreal &lenght, qreal &overlap, qreal &elapsedTime, int &totalIterations, uint &seed, RASTERVORONOIPACKING::RasterPackingSolution &solution)
#endif
{
    qDebug() << "Started";
    seed = this->seed;
    QTime myTimer; myTimer.start();
    int itNum = 0;
    int totalItNum = 0;
    int worseSolutionsCount = 0;
	bool success = false;
	int curLength = solver->getCurrentWidth();
	int minSuccessfullLength = curLength;
	qreal rdec = 0.04; qreal rinc = 0.01;
    qreal minOverlap, curOverlap;
    RASTERVORONOIPACKING::RasterPackingSolution bestSolution = threadSolution;
    solver->resetWeights();
    emit solutionGenerated(threadSolution, scale);
	int numLoops = 0;

	minOverlap = solver->getGlobalOverlap(threadSolution, parameters);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, scale);

	while (myTimer.elapsed() / 1000.0 < parameters.getTimeLimit()) {
		while (worseSolutionsCount < parameters.getNmo() && myTimer.elapsed() / 1000.0 < parameters.getTimeLimit()) {
			if(finishNow) break;
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
			if(itNum % 50 == 0) {
				emit statusUpdated(totalItNum, worseSolutionsCount, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, curLength, minSuccessfullLength);
				emit solutionGenerated(threadSolution, scale);
				emit weightsChanged();
				#ifdef CONSOLE
				std::cout << "\r" << "Lenght: " << curLength/nonzoomscale << ". Min Length: " << minSuccessfullLength/nonzoomscale << ". It: " << totalItNum << " (" << worseSolutionsCount << "). Overlap: " << curOverlap / scale << ". Min overlap: " << minOverlap / scale << ". Time: " << myTimer.elapsed() / 1000.0 << " secs.";
				#endif
			}
			itNum++; totalItNum++;
		}
		if (!parameters.isFixedLength()) {
			// Reduce or expand container
			if(success) {
				bestSolution = threadSolution; minSuccessfullLength = curLength;
				curLength = qRound((1.0 - rdec)*(qreal)solver->getCurrentWidth());
				#ifdef CONSOLE
				std::cout << "\n" << "New minimum length obtained: " << minSuccessfullLength / nonzoomscale << ". Elapsed time: " << myTimer.elapsed() / 1000.0 << " secs.\n";
				#endif
			}
			else {
				if (qRound((1.0 + rinc)*(qreal)solver->getCurrentWidth()) >= minSuccessfullLength) {
					if (numLoops > MAXLOOPSPERLENGTH) {
						solver->generateRandomSolution(threadSolution, parameters);
						numLoops = 0;
					}
					else numLoops++;
				}
				else curLength = qRound((1.0 + rinc)*(qreal)solver->getCurrentWidth());
			}
			solver->setContainerWidth(curLength, threadSolution);
			emit containerLengthChanged(curLength);
			emit solutionGenerated(threadSolution, scale);
			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution, parameters);
		}
		else {
			if (success) break;
			else {
				if (numLoops > MAXLOOPSPERLENGTH) {
					solver->generateRandomSolution(threadSolution, parameters);
					numLoops = 0;
				}
				else numLoops++;
			}
		}
		itNum = 0;
		worseSolutionsCount = 0;
		solver->resetWeights();
		emit weightsChanged();
	}
	//

    if(finishNow) qDebug() << "Aborted!";
 //   
	if (!parameters.isFixedLength()) {
		emit containerLengthChanged(minSuccessfullLength);
		finishedExecution(totalItNum, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, minSuccessfullLength);
	}
	else emit finishedExecution(totalItNum, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, solver->getCurrentWidth());
    emit solutionGenerated(bestSolution, scale);
	if (parameters.isFixedLength()) qDebug() << "\nFinished. Total iterations:" << totalItNum << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << myTimer.elapsed() / 1000.0;
	else qDebug() << "\nFinished. Total iterations:" << totalItNum << ".Minimum length =" << minSuccessfullLength / nonzoomscale << ". Elapsed time:" << myTimer.elapsed() / 1000.0;

    #ifdef CONSOLE
        overlap = minOverlap;
		lenght = minSuccessfullLength / nonzoomscale;
        elapsedTime = myTimer.elapsed()/1000.0;
        totalIterations = totalItNum;
        for(int i = 0; i < bestSolution.getNumItems(); i++) {
            solution.setPosition(i, bestSolution.getPosition(i));
            solution.setOrientation(i, bestSolution.getOrientation(i));
        }
    #endif
}
