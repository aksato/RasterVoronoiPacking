#ifndef PACKINGTHREAD_H
#define PACKINGTHREAD_H

#include <memory>
#include "rasterpackingsolution.h"
#include "rasterstrippackingsolver.h"

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
class PackingThread : public QThread
{
    Q_OBJECT
public:
    PackingThread(QObject *parent = 0);
    ~PackingThread();

    void setInitialSolution(RASTERVORONOIPACKING::RasterPackingSolution &initialSolution);
    void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> _solver) {
		solver =_solver;
		threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(solver->getNumItems());
	}
	void setParameters(RASTERVORONOIPACKING::RasterStripPackingParameters &_parameters) { parameters.Copy(_parameters); }
	//void setScale(qreal scale) { this->scale = scale; this->nonzoomscale = scale; }
	//void setScale(qreal scale, qreal nonzoomscale) { this->scale = scale; this->nonzoomscale = nonzoomscale; }
	uint getSeed() { return seed; }

protected:
	void run();

signals:
	// Basic signals
	void statusUpdated(int curLength, int totalItNum, int worseSolutionsCount, qreal curOverlap, qreal minOverlap, qreal elapsed);
	void minimumLenghtUpdated(int minLength, int totalItNum, qreal elapsed, uint seed);
	void finishedExecution(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int minLength, int totalItNum, qreal curOverlap, qreal minOverlap, qreal elapsed, uint seed);
	// GUI signals
	void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length);
	//void containerLengthChanged(int newLength);
	void weightsChanged();

	//void solutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &solution, qreal scale);
	//void statusUpdated(int totalItNum, int worseSolutionsCount, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal nonzoomscale, qreal scale, int curLength, int minLength);
	//void finishedExecution(int totalItNum, qreal  curOverlap, qreal minOverlap, qreal elapsed, qreal scale, qreal zoomScale, int minLength, uint seed);
	//void weightsChanged();
	//void containerLengthChanged(int newLength);
	//void minimumLenghtUpdated(int minLength, qreal scale, qreal zoomScale, int totalItNum, qreal elapsed, uint seed);
	//void finalSolutionGenerated(const RASTERVORONOIPACKING::RasterPackingSolution &finalSolution, qreal scale, qreal finalLength, uint seed);
	//void totallyFinished();

public slots:
	void abort();

private:
	uint seed; 
	bool m_abort; 
    std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> solver;
	RASTERVORONOIPACKING::RasterStripPackingParameters parameters;
	RASTERVORONOIPACKING::RasterPackingSolution threadSolution;
    //qreal scale, nonzoomscale;
};

#endif // PACKINGTHREAD_H
