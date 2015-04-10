#include "packingthread.h"
#include <QDebug>
#include <QTime>

#ifndef CONSOLE
PackingThread::PackingThread(QObject *parent) :
    QThread(parent)
#else
#include <iostream>
PackingThread::PackingThread(QObject *parent) :
    QObject(parent)
#endif
{
    Nmo = 200;
    maxSeconds = 60;
    heuristicType = 1;
	useCUDA = false; cacheMaps = false;
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

void PackingThread::setParameters(const int _Nmo, const int _heuristicType, const int _maxSeconds, const bool _useCUDA, const bool _cacheMaps, const bool _stripPacking) {
    this->Nmo = _Nmo;
    this->heuristicType = _heuristicType;
    this->maxSeconds = _maxSeconds;
	this->useCUDA = _useCUDA;
	this->cacheMaps = _cacheMaps;
	this->stripPacking = _stripPacking;
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
void PackingThread::run(qreal &overlap, qreal &elapsedTime, int &totalIterations, uint &seed, RASTERVORONOIPACKING::RasterPackingSolution &solution)
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

    bool gls = (heuristicType != 0);

	int numLoops = 0;

	if (heuristicType == 2) solver->switchProblem(true);
	minOverlap = solver->getGlobalOverlap(threadSolution);
	itNum++; totalItNum++;
	emit solutionGenerated(threadSolution, scale);

	while (myTimer.elapsed() / 1000.0 < maxSeconds) {
		while (worseSolutionsCount < Nmo && myTimer.elapsed() / 1000.0 < maxSeconds) {
			if(finishNow) break;
			if (heuristicType == 0 || heuristicType == 1) {
				if (useCUDA) solver->performLocalSearchGPU(threadSolution, gls);
				else if(cacheMaps) solver->performLocalSearchwithCache(threadSolution, gls);
				else solver->performLocalSearch(threadSolution, gls);
			}
			else if(heuristicType == 2) solver->performTwoLevelLocalSearch(threadSolution, gls, 3);
			emit solutionGenerated(threadSolution, scale);
			if(heuristicType == 2) solver->switchProblem(true);
			if(heuristicType != 0) solver->updateWeights(threadSolution);
			//emit weightsChanged();
			if(heuristicType == 2) solver->switchProblem(true);
			curOverlap = solver->getGlobalOverlap(threadSolution);
			if(curOverlap < minOverlap) {
				minOverlap = curOverlap;
				worseSolutionsCount = 0;
				if(!stripPacking) bestSolution = threadSolution;
			}
			else worseSolutionsCount++;
			//if (qFuzzyCompare(1.0 + 0.0, 1.0 + minOverlap)) { success = true; break; }
			if (qFuzzyCompare(1.0 + 0.0, 1.0 + curOverlap)) { success = true; break; }
			if(itNum % 50 == 0) {
				emit statusUpdated(totalItNum, worseSolutionsCount, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, curLength, minSuccessfullLength);
				emit weightsChanged();
				#ifdef CONSOLE
					std::cout << "\r" << "It: " << totalItNum << " (" << worseSolutionsCount
							 <<  "). Overlap: " << curOverlap/scale << ". Min overlap: "
							  << minOverlap/scale << ". Time: " << myTimer.elapsed()/1000.0 <<  " secs.";
				#endif
			}
			itNum++; totalItNum++;

			//if(worseSolutionsCount == Nmo) {
			//    itNum = 0;
			//    worseSolutionsCount = 0;
			//    solver->resetWeights();
			//	  emit weightsChanged();
			//}
		}
		if (stripPacking) {
			// Reduce or expand container
			if (success) {
				bestSolution = threadSolution; minSuccessfullLength = curLength;
				curLength = qRound((1.0 - rdec)*(qreal)solver->getCurrentWidth());
			}
			else {
				//curLength = qRound((1.0 + rinc)*(qreal)solver->getCurrentWidth());
				//if (curLength >= minSuccessfullLength) {
				//	curLength = qRound((1.0 - rdec)*(qreal)solver->getCurrentWidth());
				//	solver->generateRandomSolution(threadSolution);
				//}
				if (qRound((1.0 + rinc)*(qreal)solver->getCurrentWidth()) >= minSuccessfullLength) {
					if (numLoops > 5) solver->generateRandomSolution(threadSolution);
					numLoops++;
				}
				else curLength = qRound((1.0 + rinc)*(qreal)solver->getCurrentWidth());
			}
			solver->setContainerWidth(curLength, threadSolution);
			emit containerLengthChanged(curLength);
			emit solutionGenerated(threadSolution, scale);
			success = false;
			minOverlap = solver->getGlobalOverlap(threadSolution);
		}
		itNum = 0;
		worseSolutionsCount = 0;
		solver->resetWeights();
		emit weightsChanged();
	}
	

    if(finishNow) qDebug() << "Aborted!";
    
	if (stripPacking) {
		emit containerLengthChanged(minSuccessfullLength);
		finishedExecution(totalItNum, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, minSuccessfullLength);
	}
	else emit finishedExecution(totalItNum, curOverlap, minOverlap, myTimer.elapsed() / 1000.0, this->scale, solver->getCurrentWidth());
    emit solutionGenerated(bestSolution, scale);
    qDebug() << "\nFinished. Total iterations:" << totalItNum << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << myTimer.elapsed()/1000.0;

    #ifdef CONSOLE
        overlap = minOverlap;
        elapsedTime = myTimer.elapsed()/1000.0;
        totalIterations = totalItNum;
        for(int i = 0; i < bestSolution.getNumItems(); i++) {
            solution.setPosition(i, bestSolution.getPosition(i));
            solution.setOrientation(i, bestSolution.getOrientation(i));
        }
    #endif
}
