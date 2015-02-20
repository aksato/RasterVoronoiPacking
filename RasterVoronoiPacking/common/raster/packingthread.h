#ifndef PACKINGTHREAD_H
#define PACKINGTHREAD_H

#include <memory>
#include "rasterpackingsolution.h"
#include "rasterstrippackingsolver.h"

#ifndef CONSOLE
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
class PackingThread : public QThread
#else
class PackingThread : public QObject
#endif
{
    Q_OBJECT
public:
    PackingThread(QObject *parent = 0);
    ~PackingThread();

    void setInitialSolution(RASTERVORONOIPACKING::RasterPackingSolution &initialSolution);
    void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> _solver) {solver =_solver;}
    void setParameters(const int _Nmo, const int _heuristicType, const int _maxSeconds);
    void setScale(qreal scale) {this->scale = scale;}

#ifndef CONSOLE
    protected:
        void run();
#else
    void run(qreal &overlap, qreal &elapsedTime, int &totalIterations, uint &seed, RASTERVORONOIPACKING::RasterPackingSolution &solution);
#endif


signals:
    void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
    void statusUpdated(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale);
    void finishedExecution(int totalItNum, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale);
    void weightsChanged();

public slots:

private:
    RASTERVORONOIPACKING::RasterPackingSolution threadSolution;
    std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
    bool finishNow;
    int Nmo, heuristicType, maxSeconds;
    uint seed;
    qreal scale;
};

#endif // PACKINGTHREAD_H
