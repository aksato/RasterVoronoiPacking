#ifndef PACKINGTHREAD_H
#define PACKINGTHREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <memory>
#include "rasterpackingsolution.h"
#include "rasterstrippackingsolver.h"

class PackingThread : public QThread
{
    Q_OBJECT
public:
    PackingThread(QObject *parent = 0);
    ~PackingThread();

    void setInitialSolution(RASTERVORONOIPACKING::RasterPackingSolution &initialSolution);
    void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> _solver) {solver =_solver;}
    void setParameters(const int _Nmo, const int _heuristicType, const int _maxSeconds);
    void setScale(qreal scale) {this->scale = scale;}

signals:
    void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
    void statusUpdated(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale);
    void finishedExecution(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale);
    void weightsChanged();

public slots:

protected:
    void run();

private:
    RASTERVORONOIPACKING::RasterPackingSolution threadSolution;
    std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
    bool finishNow;
    int Nmo, heuristicType, maxSeconds;
    qreal scale;
};

#endif // PACKINGTHREAD_H
