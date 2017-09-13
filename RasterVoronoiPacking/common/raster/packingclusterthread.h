#ifndef PACKINGCLUSTERTHREAD_H
#define PACKINGCLUSTERTHREAD_H

#include "packingthread.h"

class PackingClusterThread : public PackingThread {
	Q_OBJECT
public:
	PackingClusterThread(QObject *parent = 0) {};
	~PackingClusterThread() {};

	void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> _solver, std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> _clusterSolver) {
		PackingThread::setSolver(_clusterSolver);
		clusterSolver = _clusterSolver;
		originalSolver = _solver;
	}

signals:
	void unclustered(const RASTERVORONOIPACKING::RasterPackingSolution &solution, int length, qreal elapsed);

protected:
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> clusterSolver;
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver> originalSolver;

	void run();
};

#endif // PACKINGCLUSTERTHREAD_H
