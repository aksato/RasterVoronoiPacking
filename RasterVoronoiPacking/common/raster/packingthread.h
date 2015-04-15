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
	//void setParameters(const int _Nmo, const int _heuristicType, const int _maxSeconds, const bool _useCUDA, const bool _cacheMaps, const bool _stripPacking);
	void setParameters(RASTERVORONOIPACKING::RasterStripPackingParameters &_parameters) {parameters.Copy(_parameters);}
	void setScale(qreal scale) { this->scale = scale; this->nonzoomscale = scale; }
	void setScale(qreal scale, qreal nonzoomscale) { this->scale = scale; this->nonzoomscale = nonzoomscale; }
	//void setCuda(bool useCUDA) { this->useCUDA = useCUDA; }
	//void setCacheUse(bool cacheMaps) { this->cacheMaps = cacheMaps; }
	//void setStripPacking(bool _stripPacking) { this->stripPacking = stripPacking; }

#ifndef CONSOLE
    protected:
        void run();
#else
	void run(qreal &lenght, qreal &overlap, qreal &elapsedTime, int &totalIterations, uint &seed, RASTERVORONOIPACKING::RasterPackingSolution &solution);
#endif


signals:
    void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
    void statusUpdated(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, int curLength, int minLength);
	void finishedExecution(int totalItNum, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, int minLength);
    void weightsChanged();
	void containerLengthChanged(int newLength);

public slots:

private:
    RASTERVORONOIPACKING::RasterPackingSolution threadSolution;
    std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
    bool finishNow;
	RASTERVORONOIPACKING::RasterStripPackingParameters parameters;
 //   int Nmo, heuristicType, maxSeconds;
    uint seed;
    qreal scale, nonzoomscale;
	//bool useCUDA;
	//bool cacheMaps;
	//bool stripPacking;
};

#endif // PACKINGTHREAD_H
