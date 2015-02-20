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

void PackingThread::setParameters(const int _Nmo, const int _heuristicType, const int _maxSeconds) {
    this->Nmo = _Nmo;
    this->heuristicType = _heuristicType;
    this->maxSeconds = _maxSeconds;
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
    qreal minOverlap, curOverlap;
    RASTERVORONOIPACKING::RasterPackingSolution bestSolution = threadSolution;
    solver->resetWeights();
    emit solutionGenerated(threadSolution, scale);

    bool gls = (heuristicType != 0);
    if(heuristicType == 2) solver->switchProblem(true);
    minOverlap = solver->getGlobalOverlap(threadSolution);
//    else if(heuristicType == 2) minOverlap = solver->getGlobalZoomedOverlap(threadSolution);
    itNum++; totalItNum++;
    emit solutionGenerated(threadSolution, scale);
    while(myTimer.elapsed()/1000.0 < maxSeconds) {
        if(finishNow) break;
        if(heuristicType == 0 || heuristicType == 1) solver->performLocalSearch(threadSolution, gls);
        else if(heuristicType == 2) solver->performTwoLevelLocalSearch(threadSolution, gls, 3);
        emit solutionGenerated(threadSolution, scale);
        if(heuristicType == 2) solver->switchProblem(true);
        if(heuristicType != 0) solver->updateWeights(threadSolution);
        emit weightsChanged();
        if(heuristicType == 2) solver->switchProblem(true);
        curOverlap = solver->getGlobalOverlap(threadSolution);
        if(curOverlap < minOverlap) {
            minOverlap = curOverlap;
            worseSolutionsCount = 0;
            bestSolution = threadSolution;
        }
        else worseSolutionsCount++;
        if(qFuzzyCompare(1.0 + 0.0, 1.0 + minOverlap)) break;
        if(itNum % 50 == 0) {
            emit statusUpdated(totalItNum, worseSolutionsCount, curOverlap, minOverlap, myTimer.elapsed()/1000.0, this->scale);
            #ifdef CONSOLE
                std::cout << "\r" << "It: " << totalItNum << " (" << worseSolutionsCount
                         <<  "). Overlap: " << curOverlap/scale << ". Min overlap: "
                          << minOverlap/scale << ". Time: " << myTimer.elapsed()/1000.0 <<  " secs.";
            #endif
        }
        itNum++; totalItNum++;

        if(worseSolutionsCount == Nmo) {
            itNum = 0;
            worseSolutionsCount = 0;
            solver->resetWeights();
        }
    }
    if(finishNow) qDebug() << "Aborted!";
    emit finishedExecution(totalItNum, curOverlap, minOverlap, myTimer.elapsed()/1000.0, this->scale);
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
