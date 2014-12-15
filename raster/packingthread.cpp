#include "packingthread.h"
#include <QDebug>
#include <QTime>

PackingThread::PackingThread(QObject *parent) :
    QThread(parent)
{
    Nmo = 200;
    maxSeconds = 60;
    heuristicType = 1;
    finishNow = false;
}

PackingThread::~PackingThread() {
    finishNow = true;
    wait();
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

void PackingThread::run() {
    qDebug() << "Started";
    QTime myTimer; myTimer.start();
    int seed = QTime::currentTime().msec();// seed = 561; //TOREMOVE
    qDebug() << "Seed:" << seed;
    qsrand(seed);
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
        else if(heuristicType == 2) solver->performTwoLevelLocalSearch(threadSolution, 3, gls);
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
        if(itNum % 50 == 0) emit statusUpdated(totalItNum, worseSolutionsCount, curOverlap, minOverlap, myTimer.elapsed()/1000.0, this->scale);
        itNum++; totalItNum++;

        if(worseSolutionsCount == Nmo) {
            itNum = 0;
            worseSolutionsCount = 0;
            solver->resetWeights();
        }
    }
    if(finishNow) qDebug() << "Aborted!";
    emit finishedExecution(totalItNum, worseSolutionsCount, curOverlap, minOverlap, myTimer.elapsed()/1000.0, this->scale);
    qDebug() << "Finished. Total iterations:" << totalItNum << ".Minimum overlap =" << minOverlap << ". Elapsed time:" << myTimer.elapsed()/1000.0;
    emit solutionGenerated(bestSolution, scale);
}
