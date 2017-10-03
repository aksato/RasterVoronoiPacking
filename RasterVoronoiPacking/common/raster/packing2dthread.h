#ifndef PACKING2DTHREAD_H
#define PACKING2DTHREAD_H

#include "packingthread.h"

class Packing2DThread : public PackingThread {
	Q_OBJECT
public:
	Packing2DThread(QObject *parent = 0) {};
	~Packing2DThread() {};

	void setSolver(std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> _solver) {
		solver2d = _solver;
		threadSolution = RASTERVORONOIPACKING::RasterPackingSolution(solver2d->getNumItems());
	}

protected:
	std::shared_ptr<RASTERVORONOIPACKING::RasterStripPackingSolver2D> solver2d;

	void run();
};

#endif // PACKING2DTHREAD_H
